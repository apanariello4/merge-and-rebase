"""``sequential_impl='incremental'`` of the anchors closed-loop fit (anchors/synthesis.py ``fit_tracking``).

A tiny real open_clip ``VisionTransformer`` (attention, LayerNorms, class token) is used so that the block-only forward is
tested against the full-model capture. Incremental and resident fits must give the same corrections: max relative Frobenius
difference <= 1e-6 in float64, a looser bound in float32.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.methods._ariadne.capture import capture_tokens
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.layouts import COMPONENT_INPUT_KIND
from merge_and_rebase.rebase.methods.anchors import AnchorsConfig, parse_anchors_config
from merge_and_rebase.rebase.methods.anchors.config import validate_text_config
from merge_and_rebase.rebase.methods.anchors.synthesis import (
    IncrementalGenericInputs,
    LazyGenericInputs,
    ViTBlockRunner,
    capture_generic_inputs,
    fit_tracking,
    make_synthesis_hook,
)

DEPTH = 4
KEY = "visual.transformer.resblocks.{}.mlp.c_proj"
CFG = DirectResidualConfig(components=("mlp.c_proj",), ridge_estimator="fixed_relative", ridge_relative=1e-6, num_batches=6)


class _Clip(torch.nn.Module):
    def __init__(self, depth, batch_first=False):
        super().__init__()
        self.visual = VisionTransformer(image_size=8, patch_size=4, width=16, layers=depth, heads=2, mlp_ratio=2.0,
                                        output_dim=8)
        self.visual.transformer.batch_first = batch_first

    def encode_image(self, x):
        return self.visual(x)


def _model(depth, seed, dtype):
    torch.manual_seed(seed)
    m = _Clip(depth)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.to(dtype).eval()


def _batches(n, seed, dtype, bs=3):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(bs * n, 3, 8, 8, generator=g).to(dtype)
    return list(DataLoader(TensorDataset(images, torch.zeros(bs * n)), batch_size=bs, shuffle=False))


def _setup(dtype, n_batches=6, seed=3):
    base = _model(DEPTH, seed, dtype)
    truth = deepcopy(base)
    with torch.no_grad():
        for blk in truth.visual.transformer.resblocks:
            blk.mlp.c_proj.weight.add_(0.3 * torch.randn_like(blk.mlp.c_proj.weight))
            blk.mlp.c_proj.bias.add_(0.1 * torch.randn_like(blk.mlp.c_proj.bias))
    batches = _batches(n_batches, seed + 1, dtype)
    req = {str(j): (j, "boundary") for j in range(DEPTH)}
    b_out = capture_tokens(base, batches, req, "cpu")
    t_out = capture_tokens(truth, batches, req, "cpu")
    desired = {j: [t - b for t, b in zip(t_out[str(j)], b_out[str(j)], strict=True)] for j in range(DEPTH)}
    captured = {"target_batches": batches, "target_base_outputs_by_position": {j: b_out[str(j)] for j in range(DEPTH)}}
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    return base, sd, captured, desired


def _rel(a, b):
    return max(float((a[k] - b[k]).double().norm() / b[k].double().norm()) for k in b)


def _hook(kind, base, sd, dtype):
    """(resident hook, incremental hook) of one synthesis variant; ``None`` pair for the plain ridge."""
    if kind == "ridge":
        return None, None
    weights = {j: sd[KEY.format(j) + ".weight"] for j in range(DEPTH)}
    gen_batches = _batches(3, 99, dtype)
    if kind == "o2_base":
        gen = capture_generic_inputs(base, gen_batches, DEPTH, component="mlp.c_proj", tokens_per_block=30, seed=0,
                                     device="cpu")
        h = make_synthesis_hook(generic=gen, rho_g=0.1, base_weights=weights)
        return h, h
    lazy = LazyGenericInputs(base, gen_batches, component="mlp.c_proj", tokens_per_block=30, seed=0, device="cpu")
    res = make_synthesis_hook(generic=lazy, rho_g=0.1, base_weights=weights)
    inc_gen = IncrementalGenericInputs(gen_batches, tokens_per_block=30, seed=0)
    inc = make_synthesis_hook(generic=inc_gen, rho_g=0.1, base_weights=weights)
    inc.generic_provider = inc_gen
    return res, inc


def test_block_only_forward_reproduces_the_full_model_capture():
    for batch_first in (False, True):
        base = _model(DEPTH, 1, torch.float64)
        base.visual.transformer.batch_first = batch_first
        batches = _batches(2, 5, torch.float64)
        req = {**{f"{j}.out": (j, "boundary") for j in range(DEPTH)},
               **{f"{j}.h": (j, COMPONENT_INPUT_KIND["mlp.c_proj"]) for j in range(DEPTH)}}
        cap = capture_tokens(base, batches, req, "cpu")
        with ViTBlockRunner(base, "cpu") as runner, torch.no_grad():
            for k, batch in enumerate(batches):
                x = runner.prefix(batch[0])
                for j in range(DEPTH):
                    out, h = runner.run(j, x)
                    assert out.shape == cap[f"{j}.out"][k].shape and h.shape == cap[f"{j}.h"][k].shape
                    # the capture stores float32: compare at its precision
                    torch.testing.assert_close(out.float(), cap[f"{j}.out"][k], rtol=0, atol=1e-6)
                    torch.testing.assert_close(h.float(), cap[f"{j}.h"][k], rtol=0, atol=1e-6)
                    x = out


@pytest.mark.parametrize("variant", ["ridge", "o2_base", "o2_updated", "gain"])
@pytest.mark.parametrize("dtype,bound", [(torch.float64, 1e-6), (torch.float32, 2e-3)])
def test_incremental_matches_resident(variant, dtype, bound):
    base, sd, captured, desired = _setup(dtype)
    res_hook, inc_hook = _hook("ridge" if variant == "gain" else variant, base, sd, dtype)
    kw = {"gain": "lsq", "holdout": 2} if variant == "gain" else {}
    ref, ref_rows = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", solve_hook=res_hook, **kw)
    got, got_rows = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", solve_hook=inc_hook,
                                 sequential_impl="incremental", **kw)
    assert set(got) == set(ref)
    assert _rel(got, ref) <= bound, _rel(got, ref)
    assert [r["position"] for r in got_rows] == [r["position"] for r in ref_rows]
    for a, b in zip(got_rows, ref_rows, strict=True):
        assert a["relative_residual_before"] == pytest.approx(b["relative_residual_before"], rel=max(bound, 1e-6) * 10)
        if variant == "gain":
            assert a["gain"] == pytest.approx(b["gain"], rel=bound * 10)
    # the model is back at its base weights
    for k, v in base.state_dict().items():
        assert torch.equal(v, sd[k])


def test_incremental_with_fixed_corrections_and_subset_of_blocks():
    base, sd, captured, desired = _setup(torch.float64)
    fitted, _ = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu")
    kw = {"fixed": fitted, "gain": "lsq", "holdout": 0}
    ref, ref_rows = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", **kw)
    got, got_rows = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", sequential_impl="incremental", **kw)
    assert _rel(got, ref) <= 1e-6
    sub = {1: desired[1], 3: desired[3]}  # blocks without a desired change are passed through, not fitted
    ref, _ = fit_tracking(base, sd, captured, sub, config=CFG, device="cpu")
    got, _ = fit_tracking(base, sd, captured, sub, config=CFG, device="cpu", sequential_impl="incremental")
    assert set(got) == set(ref) and _rel(got, ref) <= 1e-6


def test_model_is_restored_to_its_own_weights_when_it_is_not_at_the_base():
    base, sd, captured, desired = _setup(torch.float32)
    with torch.no_grad():
        base.visual.transformer.resblocks[2].mlp.c_proj.weight.add_(1.0)
    before = {k: v.detach().clone() for k, v in base.state_dict().items()}
    for impl in ("resident", "incremental"):
        fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", sequential_impl=impl)
        assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())


def test_default_is_resident_and_unchanged():
    d = AnchorsConfig()
    assert d.sequential_impl == "resident" and "sequential_impl" not in d.to_dict()
    assert "sequential_impl" not in parse_anchors_config({"depth_fit": "tracking"}).to_dict()
    inc = parse_anchors_config({"depth_fit": "tracking", "sequential_impl": "incremental"})
    assert inc.to_dict()["sequential_impl"] == "incremental"
    base, sd, captured, desired = _setup(torch.float32)
    explicit, _ = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu", sequential_impl="resident")
    default, _ = fit_tracking(base, sd, captured, desired, config=CFG, device="cpu")
    assert all(torch.equal(explicit[k], default[k]) for k in default)


def test_rejections():
    for bad in ({"sequential_impl": "x"}, {"sequential_impl": "incremental"},  # independent fit
                {"depth_fit": "sequential", "sequential_impl": "incremental"},  # Ariadne's streaming fit
                {"depth_fit": "spread", "measure_depths": [1.0], "sequential_impl": "incremental"}):
        with pytest.raises(ValueError):
            parse_anchors_config(bad)
    parse_anchors_config({"depth_fit": "sequential", "synthesis": "ridge_preserve", "sequential_impl": "incremental"})
    with pytest.raises(ValueError, match="sequential_impl"):
        validate_text_config(AnchorsConfig(anchor_pool="wikitext103", sequential_impl="incremental"))
    # a model that is not an open_clip ViT (decoder-like) is refused by the fit itself
    with pytest.raises(ValueError, match="open_clip"):
        ViTBlockRunner(torch.nn.Linear(2, 2), "cpu")


def test_aliased_base_state_is_restored():
    """Review finding: a base dict that shares storage with the live CPU model (to_cpu_fp32(model.state_dict()))
    must not take the corrections; the model and the dict end the fit at the base weights."""
    import torch

    from merge_and_rebase.rebase.methods.anchors.synthesis import _base_and_restore

    model = torch.nn.Linear(4, 3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    aliased = model.state_dict()  # same storage as the parameters
    state, restore = _base_and_restore(model, aliased)
    model.load_state_dict({k: v + 1.0 for k, v in state.items()})
    for k in before:
        torch.testing.assert_close(state[k], before[k], rtol=0, atol=0)
        torch.testing.assert_close(restore[k], before[k], rtol=0, atol=0)
    model.load_state_dict(restore)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
