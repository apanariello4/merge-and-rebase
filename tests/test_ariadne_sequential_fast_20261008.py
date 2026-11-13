"""``sequential_impl="block_carry"`` reproduces the reference sequential Ariadne depth fit bit for bit.

The block-carry implementation (rebase/methods/_ariadne/sequential_carry.py) re-organises the computation of
``depth_fit="sequential"`` (every model stream carries its exact block input forward; each position runs only its
own block) without touching the estimator. Pinned here, on tiny random Qwen2 decoders on CPU (float64 and float32),
against `fit_direct_residual_sequential_streaming` (``sequential_impl="reference"``) from one shared Pass A:

* the fitted task vectors are equal at hash level (sha256 over every tensor's bytes) and the diagnostic rows are
  equal, for right and left padding, both bias modes (missing_bias skip / exact_form false and materialize /
  exact_form true, plus materialize / exact_form false), extension, reduction and same-depth pairings, a
  non-monotone pairing and a source model that is the target object;
* fewer blocks are executed (counted with a forward pre-hook on every decoder layer's MLP);
* the target model's state and modules are restored; config validation, serialization and method dispatch.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    parse_direct_residual_config,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.sequential_carry import (
    _BlockRangeRunner,
    fit_direct_residual_sequential_block_carry,
)
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_sequential_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.methods.ariadne import AriadneRebase
from merge_and_rebase.rebase.model_families import infer_family

TEXTS = [*CALIB_TEXTS, "a b", "the dog", *CALIB_TEXTS[:4]]  # 12 texts of uneven length -> padded batches


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _text_loader(padding_side, max_length=12):
    tok = local_tokenizer(CALIB_TEXTS, padding_side=padding_side)

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(TEXTS), batch_size=2, shuffle=False, collate_fn=collate)


def _sha(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors):
        h.update(key.encode())
        h.update(tensors[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _state_sha(model):
    return _sha(dict(model.state_dict()))


class _BlockCounter:
    """Counts executed decoder blocks (a pre-hook on each layer's MLP fires only when the block really runs).

    The hook is a plain closure, so a deep copy of a hooked model (the reference's pristine target copy) keeps
    counting into the same tally.
    """

    def __init__(self, *models):
        self._tally = [0]
        self.handles = []
        tally = self._tally

        def hit(_module, _inputs):
            tally[0] += 1

        seen = set()
        for model in models:
            if id(model) in seen:
                continue
            seen.add(id(model))
            for layer in model.model.layers:
                self.handles.append(layer.mlp.register_forward_pre_hook(hit))

    @property
    def count(self):
        return self._tally[0]

    @count.setter
    def count(self, value):
        self._tally[0] = value

    def remove(self):
        for handle in self.handles:
            handle.remove()


def _setup(source_layers, target_layers, *, dtype, missing_bias, same_object=False):
    src = tiny_qwen2(layers=source_layers, hidden=16, inter=48, seed=0).to(dtype)
    tgt = src if same_object else tiny_qwen2(layers=target_layers, seed=2).to(dtype)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    fam = infer_family(tgt)
    base_sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    if missing_bias == "materialize":
        materialize_missing_projection_biases(tgt, base_sd, family_adapter=fam, components=("mlp.c_proj",))
    return src, ft, tgt, fam, base_sd


def _fit_pair(
    source_layers,
    target_layers,
    *,
    dtype=torch.float64,
    padding_side="right",
    missing_bias="skip",
    exact_form=False,
    pairing=None,
    same_object=False,
    exclude_first_token=False,
):
    src, ft, tgt, fam, base_sd = _setup(
        source_layers, target_layers, dtype=dtype, missing_bias=missing_bias, same_object=same_object
    )
    params = {
        "depth_fit": "sequential",
        "missing_bias": missing_bias,
        "exact_form": exact_form,
        "num_batches": len(TEXTS) // 2,
    }
    cfg_ref = resolve_ariadne_decoder_config(params, fam)
    cfg_new = resolve_ariadne_decoder_config({**params, "sequential_impl": "block_carry"}, fam)
    pairing = pairing or DiscreteLayerPairing.compute(source_layers, target_layers)
    loader = _text_loader(padding_side)
    prepared = prepare_direct_residual_streaming(
        src, tgt, loader, loader, pairing, num_batches=cfg_ref.num_batches, seed=cfg_ref.seed, device="cpu",
        family_adapter=fam, exclude_first_token=exclude_first_token,
    )  # fmt: skip
    assert prepared["calibration"]["pad_row_fraction"] > 0.2  # padded batches
    args = (tgt, base_sd, src, ft, prepared, pairing)
    before = _state_sha(tgt)

    counter = _BlockCounter(tgt, src, ft)
    ref = fit_direct_residual_sequential_streaming(*args, config=cfg_ref, device="cpu", family_adapter=fam)
    ref_blocks = counter.count
    counter.count = 0
    runner = _BlockRangeRunner(fam, "cpu")
    new = fit_direct_residual_sequential_block_carry(
        *args, config=cfg_new, device="cpu", family_adapter=fam, runner=runner
    )
    new_blocks = counter.count
    counter.remove()
    assert _state_sha(tgt) == before
    assert all("forward" not in layer.__dict__ for model in (tgt, src, ft) for layer in model.model.layers)
    return ref, new, ref_blocks, new_blocks, runner, prepared


def _assert_identical(ref, new):
    (tau_ref, rows_ref), (tau_new, rows_new) = ref, new
    assert set(tau_ref) == set(tau_new)
    for key in tau_ref:
        assert tau_ref[key].dtype == tau_new[key].dtype and tau_ref[key].shape == tau_new[key].shape, key
        assert torch.equal(tau_ref[key], tau_new[key]), (key, float((tau_ref[key] - tau_new[key]).abs().max()))
    assert _sha(tau_ref) == _sha(tau_new)
    assert len(rows_ref) == len(rows_new)
    for a, b in zip(rows_ref, rows_new, strict=True):
        assert set(a) == set(b)
        for field in a:
            if isinstance(a[field], torch.Tensor):
                assert torch.equal(a[field], b[field]), field
            else:
                assert a[field] == b[field], field


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize(
    "missing_bias, exact_form",
    [("skip", False), ("materialize", True), ("materialize", False)],
    ids=["skip", "materialize_exact", "materialize_inexact"],
)
@pytest.mark.parametrize(
    "source_layers, target_layers", [(2, 3), (3, 2), (2, 4), (3, 3)], ids=["ext", "red", "ext2x", "same"]
)
def test_block_carry_bitwise_equals_reference(source_layers, target_layers, missing_bias, exact_form, padding_side):
    ref, new, ref_blocks, new_blocks, runner, _ = _fit_pair(
        source_layers, target_layers, padding_side=padding_side, missing_bias=missing_bias, exact_form=exact_form
    )
    _assert_identical(ref, new)
    # Non-vacuous: the later blocks see a nonzero upstream effect and the fit is not trivial.
    assert ref[1][-1]["upstream_effect_norm"] > 0.0
    assert any(float(v.abs().max()) > 0 for v in ref[0].values())
    if missing_bias == "materialize":
        assert any(k.endswith("down_proj.bias") for k in new[0])
    else:
        assert not any(k.endswith("down_proj.bias") for k in new[0])
    # Work: the reference runs 4 full backbones per position and batch; block_carry runs block j three times
    # (current, pristine, current after the mount; two at the last position) plus each source block once.
    n_batches = len(TEXTS) // 2
    assert ref_blocks == n_batches * target_layers * (2 * target_layers + 2 * source_layers)
    expected_new = n_batches * (3 * target_layers - 1 + 2 * source_layers)
    assert runner.executed_blocks == new_blocks == expected_new
    assert new_blocks < ref_blocks


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize(
    "missing_bias, exact_form", [("skip", False), ("materialize", True)], ids=["skip", "materialize"]
)
def test_block_carry_with_exclude_first_token_equals_reference(missing_bias, exact_form, padding_side):
    """The row mask (padding + first token) reaches the replayed target batches: block_carry stays bit-identical."""
    ref, new, _, _, _, prepared = _fit_pair(
        2, 4, padding_side=padding_side, missing_bias=missing_bias, exact_form=exact_form, exclude_first_token=True
    )
    _assert_identical(ref, new)
    assert prepared["calibration"]["n_rows_excluded_first_token"] == len(TEXTS)
    off, _, _, _, _, prepared_off = _fit_pair(
        2, 4, padding_side=padding_side, missing_bias=missing_bias, exact_form=exact_form
    )
    assert "n_rows_excluded_first_token" not in prepared_off["calibration"]
    assert _sha(off[0]) != _sha(ref[0])  # the exclusion is not a no-op for the fit


def test_block_carry_bitwise_float32():
    ref, new, ref_blocks, new_blocks, _, _ = _fit_pair(
        2, 4, dtype=torch.float32, missing_bias="materialize", exact_form=True
    )
    _assert_identical(ref, new)
    assert new_blocks < ref_blocks


def test_block_carry_non_monotone_pairing_restarts_source_stream():
    pairing = DiscreteLayerPairing(source_depth=3, target_depth=4, pairing=(2, 0, 1, 1))
    ref, new, ref_blocks, new_blocks, runner, _ = _fit_pair(3, 4, pairing=pairing)
    _assert_identical(ref, new)
    # Source blocks per stream and batch: 0..2, restart 0..0, 1..1, reuse -> 5.
    n_batches = len(TEXTS) // 2
    assert new_blocks == n_batches * (3 * 4 - 1 + 2 * 5)


def test_block_carry_source_is_target_object():
    ref, new, *_ = _fit_pair(3, 3, same_object=True)
    _assert_identical(ref, new)


def test_block_carry_single_block():
    ref, new, *_ = _fit_pair(2, 1)
    _assert_identical(ref, new)


def test_block_carry_detects_mutated_base():
    src, ft, tgt, fam, base_sd = _setup(2, 3, dtype=torch.float64, missing_bias="skip")
    cfg = resolve_ariadne_decoder_config(
        {"depth_fit": "sequential", "sequential_impl": "block_carry", "missing_bias": "skip", "exact_form": False,
         "num_batches": 6},
        fam,
    )  # fmt: skip
    pairing = DiscreteLayerPairing.compute(2, 3)
    loader = _text_loader("right")
    prepared = prepare_direct_residual_streaming(
        src, tgt, loader, loader, pairing, num_batches=6, seed=cfg.seed, device="cpu", family_adapter=fam
    )
    before = _state_sha(tgt)
    bad = {k: v.clone() for k, v in base_sd.items()}
    key = "model.layers.0.self_attn.q_proj.weight"  # (embed_tokens is tied to lm_head: a load would undo it)
    bad[key] = bad[key] + 1.0
    with pytest.raises(RuntimeError, match="not the native base"):
        fit_direct_residual_sequential_block_carry(
            tgt, bad, src, ft, prepared, pairing, config=cfg, device="cpu", family_adapter=fam
        )
    assert _state_sha(tgt) == before
    assert all("forward" not in layer.__dict__ for layer in tgt.model.layers)


def test_method_dispatch_block_carry_matches_reference():
    src, ft, tgt, fam, base_sd = _setup(2, 3, dtype=torch.float64, missing_bias="skip")
    pairing = DiscreteLayerPairing.compute(2, 3)
    loader = _text_loader("right")
    params = {"depth_fit": "sequential", "missing_bias": "skip", "exact_form": False, "num_batches": 6}
    out = {}
    for impl in ("reference", "block_carry"):
        cfg = resolve_ariadne_decoder_config({**params, "sequential_impl": impl}, fam)
        prep = AriadneRebase().prepare(
            source_base_model=src,
            source_ft_model=ft,
            target_model=tgt,
            target_base_sd=base_sd,
            source_loader=loader,
            target_loader=loader,
            pairing=pairing,
            config=cfg,
            device="cpu",
            family_adapter=fam,
        )
        out[impl] = _sha(dict(prep.task_vector))
    assert out["reference"] == out["block_carry"]


def test_sequential_impl_config():
    base = {"activation_storage": "streaming", "depth_fit": "sequential", "components": ("mlp.c_proj",)}
    assert parse_direct_residual_config(base).sequential_impl == "reference"
    assert "sequential_impl" not in direct_residual_config_dict(parse_direct_residual_config(base))
    cfg = parse_direct_residual_config({**base, "sequential_impl": "block_carry"})
    assert direct_residual_config_dict(cfg)["sequential_impl"] == "block_carry"
    with pytest.raises(ValueError, match="sequential_impl must be"):
        parse_direct_residual_config({**base, "sequential_impl": "bogus"})
    with pytest.raises(ValueError, match="requires depth_fit='sequential'"):
        parse_direct_residual_config({"sequential_impl": "block_carry"})


def test_block_carry_is_decoder_only():
    cfg = parse_direct_residual_config(
        {"activation_storage": "streaming", "depth_fit": "sequential", "components": ("mlp.c_proj",),
         "sequential_impl": "block_carry"}
    )  # fmt: skip
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="decoder-only"):
        fit_direct_residual_sequential_block_carry(
            model, {}, model, model, {}, DiscreteLayerPairing.compute(1, 1), config=cfg, device="cpu"
        )
