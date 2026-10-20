"""Decoder (HF Qwen2 toy) tests for the opt-in Ariadne ``block_target="increment"``.

(a) ``block_target="total"`` is hash-identical to leaving it unset on the decoder streaming path (calibration batches
padded); (b) the increment desired rows ``[(f_i - b_i) - (f_p - b_p)] Q_j / g(j)`` recomputed from plain hooks, on
CONTENT rows only (padding excluded identically for both difference terms), match the streaming fit; (c) the LLM
table-3 rerun seed config style (preset ariadne, materialize bias, fixed relative
ridge, position chunk) resolves through the decoder resolver with ``block_target="increment"``, and the real Qwen2.5
0.5B <-> 1.5B depth pairings are non-decreasing.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_qwen2
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.rebase.config_schema import canonicalize
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY
from merge_and_rebase.rebase.methods._ariadne.config import increment_references, resolve_ariadne_decoder_config
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)
from merge_and_rebase.rebase.model_families import infer_family
from merge_and_rebase.rebase.registry import get_method

TEXTS = CALIB_TEXTS * 2
REGIMES = [pytest.param(2, 3, id="extend"), pytest.param(3, 2, id="shrink")]


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _loader(side="right", max_length=12):
    tok = local_tokenizer(CALIB_TEXTS, padding_side=side)

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(TEXTS), batch_size=2, shuffle=False, collate_fn=collate)


def _models(source_depth, target_depth):
    src = tiny_qwen2(layers=source_depth, seed=0)
    return src, perturbed_copy(src, scale=0.05, seed=1), tiny_qwen2(layers=target_depth, seed=2)


def _cfg(**kw):
    base = dict(
        num_batches=len(TEXTS) // 2,
        components=("mlp.c_proj",),
        activation_storage="streaming",
        ridge_estimator="empirical_bayes",
        alignment_map="ridge",
        missing_bias="skip",
        exact_form=False,
    )
    return DirectResidualConfig(**{**base, **kw})


def _sha(tv):
    h = hashlib.sha256()
    for k in sorted(tv):
        h.update(k.encode())
        h.update(tv[k].detach().double().contiguous().numpy().tobytes())
    return h.hexdigest()


def _prepare(src, ft, tgt, source_depth, target_depth, cfg, side="right"):
    return get_method("ariadne").prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd={k: v.clone() for k, v in tgt.state_dict().items()},
        source_loader=_loader(side),
        target_loader=_loader(side),
        pairing=DiscreteLayerPairing.compute(source_depth, target_depth),
        config=cfg,
        device="cpu",
        family_adapter=infer_family(src),
    )


# (a) total == unset, hash level, padded decoder batches -------------------------------------------------------------


@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_total_hash_identical_to_unset_decoder(source_depth, target_depth):
    src, ft, tgt = _models(source_depth, target_depth)
    default = _prepare(src, ft, tgt, source_depth, target_depth, _cfg())
    explicit = _prepare(src, ft, tgt, source_depth, target_depth, _cfg(block_target="total"))
    inc = _prepare(src, ft, tgt, source_depth, target_depth, _cfg(block_target="increment"))
    assert _sha(default.task_vector) == _sha(explicit.task_vector)
    assert _sha(default.task_vector) != _sha(inc.task_vector)
    assert all("block_target" not in r for r in default.diagnostics + explicit.diagnostics)
    assert all(r["block_target"] == "increment" for r in inc.diagnostics)


# (b) explicit recomputation on content rows -------------------------------------------------------------------------


class _Recorder(ResidualSufficientStatistics):
    def __init__(self, store, key, **kwargs):
        super().__init__(**kwargs)
        self._store, self._key = store, key

    def update(self, h, e, t_in, t_out, row_mask=None):
        self._store.setdefault(self._key, []).append(e.double().clone())
        super().update(h, e, t_in, t_out, row_mask)


def _boundaries(model, batches):
    """Residual stream after each decoder layer, content rows only, float64 ``[layer][N, d]``."""
    layers = model.model.layers
    out = [[] for _ in layers]
    state = {}

    def hook(k):
        def _h(_m, _i, v):
            v = v[0] if isinstance(v, tuple) else v
            out[k].append(v.detach().double()[state["mask"]])

        return _h

    handles = [layer.register_forward_hook(hook(k)) for k, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            for batch in batches:
                state["mask"] = batch[ROW_MASK_KEY]
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(o) for o in out]


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("source_depth,target_depth", REGIMES)
def test_increment_matches_explicit_recomputation_decoder(source_depth, target_depth, side):
    src, ft, tgt = _models(source_depth, target_depth)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    cfg = _cfg(block_target="increment")
    prepared = prepare_direct_residual_streaming(
        src,
        tgt,
        _loader(side),
        _loader(side),
        pairing,
        num_batches=cfg.num_batches,
        seed=cfg.seed,
        device="cpu",
        family_adapter=infer_family(src),
        source_ft_model=ft,
        alignment_map="ridge",
    )
    n_pad = sum(int((~b[ROW_MASK_KEY]).sum()) for b in prepared["source_batches"])
    assert n_pad > 0, "fixture must contain padding"
    store: dict = {}
    sd = {k: v.clone() for k, v in tgt.state_dict().items()}
    _, rows = fit_direct_residual_streaming(
        tgt,
        sd,
        src,
        ft,
        prepared,
        pairing,
        config=cfg,
        device="cpu",
        family_adapter=infer_family(src),
        stats_factory=lambda pos, comp, dev: _Recorder(store, pos, device=dev),
    )
    sb = _boundaries(src, prepared["source_batches"])
    sf = _boundaries(ft, prepared["source_batches"])
    pi = pairing.pairing
    refs = increment_references(pi)
    for j in range(target_depth):
        g = sum(1 for k in pi if k == pi[j])
        delta = sf[pi[j]] - sb[pi[j]]
        if refs[j] is not None:
            delta = delta - (sf[pi[refs[j]]] - sb[pi[refs[j]]])
        want = delta @ prepared["q_by_position"][j].double() / g
        got = torch.cat(store[j])
        assert got.shape == want.shape, "padding rows must be absent from the fit rows"
        rel = float((got - want).norm() / want.norm())
        assert rel < 1e-5, f"j={j} rel={rel}"
    assert {r["position"] for r in rows} == set(range(target_depth))


# (c) the LLM config resolves with block_target='increment' ----------------------------------------------------------

_CONFIG_PARAMS = {
    "exact_form": True,
    "missing_bias": "materialize",
    "num_batches": 100,
    "preset": "ariadne",
    "ridge_estimator": "fixed_relative",
    "ridge_relative": 0.01,
    "seed": 0,
    "streaming_position_chunk": 14,
}


def test_llm_config_resolves_with_increment():
    cfg = {
        "method": {"name": "ariadne", "params": {**_CONFIG_PARAMS, "block_target": "increment"}},
        "model_arch": "qwen",
        "model_kind": "causal_lm",
        "source_model_name_or_path": "Qwen/Qwen2.5-0.5B",
        "target_model_name_or_path": "Qwen/Qwen2.5-1.5B",
    }
    params = dict(canonicalize(cfg)["ariadne_params"])  # what llm_rebase.context passes to the decoder resolver
    resolved = resolve_ariadne_decoder_config(params, infer_family(tiny_qwen2()))
    assert resolved.block_target == "increment"
    assert (resolved.activation_storage, resolved.depth_fit, resolved.missing_bias) == (
        "streaming",
        "independent",
        "materialize",
    )
    unset = resolve_ariadne_decoder_config(_CONFIG_PARAMS, infer_family(tiny_qwen2()))
    assert unset.block_target == "total"


@pytest.mark.parametrize("source_depth,target_depth", [(24, 28), (28, 24)])
def test_qwen25_pairings_are_non_decreasing(source_depth, target_depth):
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth).pairing
    refs = increment_references(pairing)
    assert len(refs) == target_depth and refs[0] is None
