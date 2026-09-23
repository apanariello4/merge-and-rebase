"""Tests for Direct Residual's ``procrustes_source='gradient'`` mode.

``DirectResidualConfig.procrustes_source`` (default ``"activation"``, bit-
identical to pre-ablation code) changes ONLY the statistic the per-position
Procrustes alignment ``Q_j`` is fit on: ``"gradient"`` fits it on block-
boundary GRADIENTS ``dL/dT_i`` (source base) / ``dL/dT_j`` (target base),
``L`` = BiCo's own zero-shot contrastive CE
(``models.grad_recipes.clip_contrastive_recipe``), instead of the forward
ACTIVATION banks. ``D_j = (S_ft - S_base) @ Q_j`` is unchanged in form in
both modes -- a gradient difference is never used as the regression target,
only as the alignment statistic.

Reuses the real ``open_clip.transformer.VisionTransformer`` fixture pattern
from ``tests/test_direct_residual_open_clip_integration.py`` (offline,
random init) rather than the hand-built toy fixture, since gradient capture
needs a real ``nn.MultiheadAttention``/``ln_1``/GELU coupling to be a
meaningful check.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy

import pytest
import torch
from open_clip.transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.direct_residual import (
    DirectResidualConfig,
    capture_paired_boundary_activations,
    compute_desired_effects,
    fit_direct_residual,
    parse_direct_residual_config,
)
from merge_and_rebase.eval.target_informed_runtime import _aligned, _rows, capture_block_gradients
from merge_and_rebase.eval.target_residual_completion import centered_rectangular_procrustes
from merge_and_rebase.models.grad_recipes import clip_contrastive_recipe
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")),
]


# --------------------------------------------------------------------------
# Fixtures: same shapes as test_direct_residual_open_clip_integration.py.
# --------------------------------------------------------------------------


class _CLIPLike(torch.nn.Module):
    """Minimal ``open_clip``-shaped wrapper carrying ``logit_scale`` too, so
    ``clip_contrastive_recipe`` (which reads ``model.logit_scale.exp()``) can
    run against it, exactly as it does against a real ``open_clip.CLIP``."""

    def __init__(self, visual: VisionTransformer):
        super().__init__()
        self.visual = visual
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def encode_image(self, x):
        return self.visual(x)


class _DummyClassifier:
    """Stands in for ``OpenClipClassifier``: only ``.normalize`` is read by
    ``clip_contrastive_recipe`` when ``text_features`` is supplied explicitly
    (the lazy ``_compute_zeroshot_text_features`` branch is never reached in
    these tests)."""

    normalize = True

    def _compute_zeroshot_text_features(self, *_args, **_kwargs):
        raise AssertionError("text_features was supposed to be provided explicitly")


def _make_vit(*, image_size, patch_size, width, layers, heads, seed):
    torch.manual_seed(seed)
    vt = VisionTransformer(
        image_size=image_size, patch_size=patch_size, width=width, layers=layers, heads=heads,
        mlp_ratio=2.0, ls_init_value=None, output_dim=width, pool_type="tok",
    )
    return _CLIPLike(vt).eval()


def _tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


class _IdentityTensorDataset(TensorDataset):
    def __init__(self, images, labels, sample_ids):
        super().__init__(images, labels)
        self.sample_ids = sample_ids


_N = 6  # samples; also the number of pseudo-classes (labels = arange(N)).


def _loader(image_size, n=_N, seed=0, sample_ids=None):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, image_size, image_size, generator=generator)
    labels = torch.arange(n)
    ids = sample_ids if sample_ids is not None else [str(i) for i in range(n)]
    return DataLoader(_IdentityTensorDataset(images, labels, ids), batch_size=2, shuffle=False)


_DIRECTIONS = {
    "extend": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                   target=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3)),
    "shrink": dict(source=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
                   target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
    "same_arch": dict(source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
                       target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2)),
}


def _direction_setup(direction, seed=101):
    spec = _DIRECTIONS[direction]
    source_base = _make_vit(seed=seed, **spec["source"])
    target_base = _make_vit(seed=seed + 1, **spec["target"])
    source_ft = _tuned_copy(source_base, seed=seed + 2)
    shared_ids = [str(i) for i in range(_N)]
    source_loader = _loader(spec["source"]["image_size"], seed=seed + 3, sample_ids=shared_ids)
    target_loader = _loader(spec["target"]["image_size"], seed=seed + 4, sample_ids=shared_ids)
    pairing = DiscreteLayerPairing.compute(spec["source"]["layers"], spec["target"]["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    source_text_feats = torch.randn(_N, spec["source"]["width"])
    target_text_feats = torch.randn(_N, spec["target"]["width"])
    return (
        source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd,
        source_text_feats, target_text_feats,
    )


def _recipes(source_text_feats, target_text_feats):
    source_recipe = clip_contrastive_recipe(
        _DummyClassifier(), [], None, text_features=source_text_feats, device="cpu",
    )
    target_recipe = clip_contrastive_recipe(
        _DummyClassifier(), [], None, text_features=target_text_feats, device="cpu",
    )
    return source_recipe, target_recipe


def _state_dict_sha256(d) -> str:
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(key.encode())
        h.update(d[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _requires_grad_flags(model):
    return {name: p.requires_grad for name, p in model.named_parameters()}


# --------------------------------------------------------------------------
# (d) capture_block_gradients never mutates parameters and restores
#     requires_grad/training/device, even when the model enters frozen.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_capture_block_gradients_restores_state_and_requires_grad(direction):
    (source_base, _ft, _tb, source_loader, _tl, pairing, _sd, source_text_feats, _tf) = _direction_setup(direction)
    source_recipe, _ = _recipes(source_text_feats, source_text_feats)
    batches = list(source_loader)

    for p in source_base.parameters():
        p.requires_grad_(False)
    frozen_flags = _requires_grad_flags(source_base)
    assert not any(frozen_flags.values())

    before_hash = _state_dict_sha256(source_base.state_dict())
    before_training = source_base.training

    requests = {str(i): i for i in range(pairing.source_depth)}
    grads = capture_block_gradients(source_base, batches, requests, source_recipe, "cpu")

    assert set(grads) == set(requests)
    for _key, values in grads.items():
        assert len(values) == len(batches)
        for tensor in values:
            assert torch.isfinite(tensor).all()

    after_hash = _state_dict_sha256(source_base.state_dict())
    assert before_hash == after_hash, "capture_block_gradients must not mutate parameter values"
    assert _requires_grad_flags(source_base) == frozen_flags, "requires_grad flags must be restored"
    assert source_base.training == before_training
    for p in source_base.parameters():
        assert p.grad is None, "no parameter gradient may be retained"


def test_capture_block_gradients_rejects_family_adapter():
    (source_base, _ft, _tb, source_loader, _tl, _pairing, _sd, source_text_feats, _tf) = _direction_setup("same_arch")
    source_recipe, _ = _recipes(source_text_feats, source_text_feats)
    batches = list(source_loader)
    with pytest.raises(NotImplementedError):
        capture_block_gradients(source_base, batches, {"0": 0}, source_recipe, "cpu", family_adapter=object())


# --------------------------------------------------------------------------
# (b) captured gradients equal torch.autograd.grad of the recipe's own loss
#     w.r.t. each block's raw output tensor, brute force.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_captured_gradients_match_autograd_brute_force(direction):
    (source_base, _ft, _tb, source_loader, _tl, pairing, _sd, source_text_feats, _tf) = _direction_setup(direction)
    source_recipe, _ = _recipes(source_text_feats, source_text_feats)
    batches = list(source_loader)
    requests = {str(i): i for i in range(pairing.source_depth)}

    captured = capture_block_gradients(source_base, batches, requests, source_recipe, "cpu")

    resblocks = source_base.visual.transformer.resblocks
    for batch_idx, batch in enumerate(batches):
        raw_outputs = {}
        handles = []
        for name, index in requests.items():
            def hook(_m, _inp, out, *, name=name, _sink=raw_outputs):
                out.retain_grad()
                _sink[name] = out
            handles.append(resblocks[index].register_forward_hook(hook))
        try:
            source_base.zero_grad(set_to_none=True)
            for p in source_base.parameters():
                p.requires_grad_(True)
            loss, _ = source_recipe(source_base, batch)
            loss.backward()
        finally:
            for h in handles:
                h.remove()
            source_base.zero_grad(set_to_none=True)

        batch_size = batch[0].shape[0]
        for name in requests:
            brute = raw_outputs[name].grad
            assert brute is not None
            from merge_and_rebase.rebase.methods.theseus import _to_tokens

            brute_tokens = _to_tokens(brute.detach(), batch_size=batch_size)
            torch.testing.assert_close(captured[name][batch_idx], brute_tokens.float(), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# (c) Q_gradient equals centered_rectangular_procrustes on the brute-force
#     (independently recomputed) gradient rows, wired end to end through
#     capture_paired_boundary_activations / compute_desired_effects.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
def test_gradient_desired_effects_match_manual_recomputation(direction):
    (
        source_base, source_ft, target_base, source_loader, target_loader, pairing, _sd,
        source_text_feats, target_text_feats,
    ) = _direction_setup(direction)
    source_recipe, target_recipe = _recipes(source_text_feats, target_text_feats)

    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=3, seed=None, device="cpu",
        procrustes_source="gradient", source_recipe=source_recipe, target_recipe=target_recipe,
    )
    diagnostics: dict[int, dict] = {}
    desired = compute_desired_effects(captured, pairing, procrustes_source="gradient", diagnostics_out=diagnostics)

    # Manual recomputation: independently re-derive Q_j and D_j from the SAME
    # captured banks, without going through compute_desired_effects, and
    # compare bit-exact.
    source_base_act = captured["source_base_outputs"]
    source_ft_act = captured["source_ft_outputs"]
    target_act = captured["target_base_outputs_by_position"]
    source_grad = captured["source_base_gradients"]
    target_grad = captured["target_base_gradients"]

    for j in range(pairing.target_depth):
        i = pairing.pairing[j]
        aligned_grad = _aligned(source_grad[i], target_grad[j])
        q, _mu_s, _mu_t = centered_rectangular_procrustes(_rows(aligned_grad).double(), _rows(target_grad[j]).double())
        q = q.float()
        base_batches = _aligned(source_base_act[i], target_act[j])
        ft_batches = _aligned(source_ft_act[i], target_act[j])
        expected = [(f - b) @ q for b, f in zip(base_batches, ft_batches, strict=True)]
        for e, d in zip(expected, desired[j], strict=True):
            torch.testing.assert_close(e, d, rtol=0, atol=0)
        assert diagnostics[j]["procrustes_source"] == "gradient"
        assert diagnostics[j]["procrustes_rank"] >= 1
        assert math.isfinite(diagnostics[j]["activation_gradient_procrustes_overlap"])


def test_gradient_and_activation_desired_effects_generically_differ():
    """Sanity: the statistic actually changed something (not a silent no-op)."""
    (
        source_base, source_ft, target_base, source_loader, target_loader, pairing, _sd,
        source_text_feats, target_text_feats,
    ) = _direction_setup("extend")
    source_recipe, target_recipe = _recipes(source_text_feats, target_text_feats)

    captured_grad = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=3, seed=None, device="cpu",
        procrustes_source="gradient", source_recipe=source_recipe, target_recipe=target_recipe,
    )
    desired_grad = compute_desired_effects(captured_grad, pairing, procrustes_source="gradient")

    captured_act = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=3, seed=None, device="cpu",
    )
    desired_act = compute_desired_effects(captured_act, pairing)

    differs = any(
        not torch.allclose(a, b, atol=1e-6)
        for j in desired_grad
        for a, b in zip(desired_grad[j], desired_act[j], strict=True)
    )
    assert differs


# --------------------------------------------------------------------------
# (a) Default path (procrustes_source omitted / "activation") is unchanged:
#     capture_paired_boundary_activations / compute_desired_effects called
#     with and without the new kwargs produce identical results.
# --------------------------------------------------------------------------


def test_default_activation_path_unaffected_by_new_kwargs():
    (
        source_base, source_ft, target_base, source_loader, target_loader, pairing, _sd, _stf, _ttf,
    ) = _direction_setup("same_arch")

    captured_default = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=3, seed=None, device="cpu",
    )
    captured_explicit = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=3, seed=None, device="cpu", procrustes_source="activation",
    )
    assert set(captured_default) == set(captured_explicit)
    assert "source_base_gradients" not in captured_default
    assert "target_base_gradients" not in captured_default

    desired_default = compute_desired_effects(captured_default, pairing)
    desired_explicit = compute_desired_effects(captured_explicit, pairing, procrustes_source="activation")
    for j in desired_default:
        for a, b in zip(desired_default[j], desired_explicit[j], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------
# (e)/(f) end-to-end fit_direct_residual in gradient mode, finite, cpu+cuda.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink", "same_arch"])
@pytest.mark.parametrize("device", DEVICES)
def test_fit_direct_residual_gradient_mode_end_to_end_finite(direction, device):
    (
        source_base, source_ft, target_base, source_loader, target_loader, pairing, target_base_sd,
        source_text_feats, target_text_feats,
    ) = _direction_setup(direction)
    source_recipe, target_recipe = _recipes(source_text_feats, target_text_feats)
    cfg = DirectResidualConfig(
        num_batches=3, ridge_relative=0.05, components=("attn.out_proj", "mlp.c_proj"),
        procrustes_source="gradient",
    )
    before_hash = _state_dict_sha256(target_base.state_dict())
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, source_loader, target_loader, pairing,
        num_batches=cfg.num_batches, seed=cfg.seed, device=device,
        procrustes_source="gradient", source_recipe=source_recipe, target_recipe=target_recipe,
    )
    desired = compute_desired_effects(captured, pairing, procrustes_source="gradient")
    corrections, diagnostics = fit_direct_residual(
        target_base, target_base_sd, captured, desired, pairing, config=cfg, device=device,
    )
    after_hash = _state_dict_sha256(target_base.state_dict())
    assert before_hash == after_hash

    assert corrections
    for value in corrections.values():
        assert torch.isfinite(value).all()
    assert diagnostics
    for row in diagnostics:
        assert row["procrustes_source"] == "gradient"


# --------------------------------------------------------------------------
# (g) Parser: default is "activation"; invalid values rejected; incompatible
#     with component_target="output_local".
# --------------------------------------------------------------------------


def test_parser_default_procrustes_source_is_activation():
    cfg = parse_direct_residual_config(None)
    assert cfg.procrustes_source == "activation"
    cfg2 = parse_direct_residual_config({})
    assert cfg2.procrustes_source == "activation"


def test_parser_accepts_gradient():
    cfg = parse_direct_residual_config({"procrustes_source": "gradient"})
    assert cfg.procrustes_source == "gradient"


def test_parser_rejects_invalid_procrustes_source():
    with pytest.raises(ValueError):
        parse_direct_residual_config({"procrustes_source": "bogus"})


def test_parser_rejects_gradient_with_output_local():
    with pytest.raises(ValueError):
        parse_direct_residual_config(
            {"procrustes_source": "gradient", "component_target": "output_local", "components": ["mlp.c_proj"]}
        )


def test_parser_allows_gradient_with_backfit_and_joint():
    cfg_backfit = parse_direct_residual_config(
        {"procrustes_source": "gradient", "block_split": "backfit", "components": ["attn.out_proj", "mlp.c_proj"]}
    )
    assert cfg_backfit.procrustes_source == "gradient"
    cfg_joint = parse_direct_residual_config(
        {"procrustes_source": "gradient", "block_split": "joint", "components": ["attn.out_proj", "mlp.c_proj"]}
    )
    assert cfg_joint.procrustes_source == "gradient"
