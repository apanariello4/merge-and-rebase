"""Gemma-2 family adapter and ARIADNE post_mlp_norm (2026-10-19).

A Gemma-2 block writes post_ffn_norm(down_proj(h)) into the residual, N(y) = y / r * g with r = sqrt(mean(y^2) + eps),
g = 1 + w. ARIADNE's block-boundary target E must reach down_proj through that norm: 'pre_norm' fits dW h ~= r E / g,
'linearized' fits (1/r) g * (dW h) ~= E. Both drop the norm Jacobian's rank-one term (checked below to be exactly
the gap to the full linearization).
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, perturbed_copy, tiny_gemma2, tiny_llama
from torch.utils.data import DataLoader, Dataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne import DirectResidualConfig, parse_direct_residual_config
from merge_and_rebase.rebase.methods._ariadne.config import (
    direct_residual_config_dict,
    reject_decoder_only_options,
    resolve_ariadne_decoder_config,
)
from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics
from merge_and_rebase.rebase.model_families import infer_family
from merge_and_rebase.rebase.registry import get_method

TEXTS = CALIB_TEXTS * 4


class _Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.sample_ids = [f"t{i}" for i in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _loader(max_length=12):
    tok = local_tokenizer(CALIB_TEXTS)

    def collate(batch):
        return dict(tok(batch, padding="max_length", max_length=max_length, return_tensors="pt"))

    return DataLoader(_Texts(TEXTS), batch_size=2, shuffle=False, collate_fn=collate)


def _hash(tv):
    h = hashlib.sha256()
    for k in sorted(tv):
        h.update(k.encode())
        h.update(tv[k].detach().double().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---- family adapter ----------------------------------------------------------------------------------------------


def test_gemma2_family_is_registered_and_reads_its_post_mlp_norm():
    model = tiny_gemma2(layers=2)
    fam = infer_family(model)
    assert fam is not None and fam.name == "gemma2" and fam.HAS_POST_MLP_NORM
    assert fam.block_count(model) == 2
    block = fam.layers(model)[1]
    assert fam.residual_writer(block) is block.mlp.down_proj
    gain, eps = fam.post_mlp_norm(block)
    torch.testing.assert_close(gain, 1.0 + block.post_feedforward_layernorm.weight.detach().float())
    y = torch.randn(5, 32)
    rms = torch.sqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    torch.testing.assert_close(block.post_feedforward_layernorm(y), y / rms * gain)
    assert not getattr(infer_family(tiny_llama()), "HAS_POST_MLP_NORM", False)
    keys = fam.transportable_keys(model.state_dict())
    assert "model.layers.0.mlp.down_proj.weight" in keys
    assert not any("layernorm" in k or k == "model.norm.weight" for k in keys)  # (1 + w) gains: passthrough


def test_dropped_term_is_exactly_the_rank_one_part_of_the_norm_jacobian():
    block = infer_family(tiny_gemma2()).layers(tiny_gemma2())[0]
    norm = block.post_feedforward_layernorm
    gain, eps = 1.0 + norm.weight.detach().float(), float(norm.eps)
    torch.manual_seed(0)
    # Gemma2RMSNorm always computes in float32 (x.float()), so the step must sit well above float32 resolution.
    y, delta = torch.randn(7, 32, dtype=torch.float64), 1e-3 * torch.randn(7, 32, dtype=torch.float64)
    actual = (norm(y + delta) - norm(y)).detach().double()
    r = torch.sqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    diag_part = delta / r * gain.double()  # what both post_mlp_norm modes model
    rank_one = -(y * delta).sum(-1, keepdim=True) * y / (32 * r**3) * gain.double()
    full = diag_part + rank_one
    assert (actual - full).norm() < 1e-2 * actual.norm()  # first order: the remainder is O(step) + float32 noise
    assert (actual - diag_part).norm() > 5 * (actual - full).norm()  # the dropped term is the main gap


def test_modes_coincide_when_rms_is_constant_and_gain_is_one():
    # pre_norm: ||H X - c E||; linearized: ||(H / c) X - E|| = ||H X - c E|| / c, and the trace-relative ridge
    # scales by 1/c^2 too, so the minimisers are identical.
    torch.manual_seed(1)
    h, e, c = torch.randn(40, 16, dtype=torch.float64), torch.randn(40, 8, dtype=torch.float64), 2.5
    a, b = ResidualSufficientStatistics(), ResidualSufficientStatistics()
    a.update(h, e * c, None, torch.eye(8, dtype=torch.float64))
    b.update(h / c, e, None, torch.eye(8, dtype=torch.float64))
    xa, _ = a.solve(ridge_relative=0.01, exact_form=False)
    xb, _ = b.solve(ridge_relative=0.01, exact_form=False)
    torch.testing.assert_close(xa, xb, rtol=1e-6, atol=1e-7)


# ---- config ----------------------------------------------------------------------------------------------------


def test_config_guards():
    gemma, llama = infer_family(tiny_gemma2()), infer_family(tiny_llama())
    with pytest.raises(ValueError, match="post_mlp_norm"):
        resolve_ariadne_decoder_config({}, gemma)
    for mode in ("pre_norm", "linearized"):
        assert resolve_ariadne_decoder_config({"post_mlp_norm": mode}, gemma).post_mlp_norm == mode
    with pytest.raises(ValueError, match="post-MLP norm"):
        resolve_ariadne_decoder_config({"post_mlp_norm": "pre_norm"}, llama)
    assert resolve_ariadne_decoder_config({}, llama).post_mlp_norm is None
    with pytest.raises(ValueError):
        parse_direct_residual_config({"activation_storage": "streaming", "post_mlp_norm": "bogus"})
    with pytest.raises(ValueError):
        parse_direct_residual_config({"activation_storage": "streaming", "post_mlp_norm": "pre_norm", "depth_fit": "sequential"})
    with pytest.raises(ValueError, match="decoder-only"):
        reject_decoder_only_options(DirectResidualConfig(post_mlp_norm="pre_norm", activation_storage="streaming"))
    assert "post_mlp_norm" not in direct_residual_config_dict(DirectResidualConfig())


# ---- end-to-end fit -------------------------------------------------------------------------------------------


def _fit(mode, stats_device=None):
    src = tiny_gemma2(layers=2, seed=0)
    ft = perturbed_copy(src, scale=0.05, seed=1)
    tgt = tiny_gemma2(layers=3, seed=2)
    fam = infer_family(src)
    cfg = DirectResidualConfig(
        num_batches=len(TEXTS) // 2,
        components=("mlp.c_proj",),
        activation_storage="streaming",
        ridge_estimator="fixed_relative",
        ridge_relative=0.01,
        missing_bias="skip",
        exact_form=False,
        post_mlp_norm=mode,
        stats_device=stats_device,
    )
    return get_method("ariadne").prepare(
        source_base_model=src,
        source_ft_model=ft,
        target_model=tgt,
        target_base_sd={k: v.clone() for k, v in tgt.state_dict().items()},
        source_loader=_loader(),
        target_loader=_loader(),
        pairing=DiscreteLayerPairing.compute(2, 3),
        config=cfg,
        device="cpu",
        family_adapter=fam,
    )


@pytest.mark.parametrize("mode", ["pre_norm", "linearized"])
def test_fit_runs_is_deterministic_and_touches_only_down_proj(mode):
    a, b = _fit(mode), _fit(mode)
    assert _hash(a.task_vector) == _hash(b.task_vector)
    nonzero = {k for k, v in a.task_vector.items() if v.abs().sum() > 0}
    assert nonzero == {f"model.layers.{j}.mlp.down_proj.weight" for j in range(3)}
    assert all(torch.isfinite(v).all() for v in a.task_vector.values())
    assert _hash(_fit(mode, stats_device="cpu").task_vector) == _hash(a.task_vector)


def test_modes_give_different_fits():
    assert _hash(_fit("pre_norm").task_vector) != _hash(_fit("linearized").task_vector)


def test_gemma2_without_mode_is_refused_at_the_fit():
    with pytest.raises(ValueError, match="post_mlp_norm"):
        resolve_ariadne_decoder_config({"preset": "ariadne"}, infer_family(tiny_gemma2()))
