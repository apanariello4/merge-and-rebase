"""``BlockExtenderCore``: the shared leaves of ``BlockExtender`` and ``DecoderBlockExtender``.

Pins the two intentional differences that the shared base must not erase: the ridge fallback
(vision: the ``ridge_weight`` config field; decoder: the constant 1e-6) and the wording of the
validation messages / log prefix.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from merge_and_rebase.eval.block_extension import BlockExtender
from merge_and_rebase.eval.block_extension_llm import DecoderBlockExtender
from merge_and_rebase.rebase.block_extension.core import BlockExtenderCore
from merge_and_rebase.rebase.model_families import infer_family
from tests.golden.test_release_golden_hashes import _brace_vision_models, _llm_source_pair


def _vision(**kw):
    base, ft = _brace_vision_models(3)
    return BlockExtender(base, ft, "cpu", verbose=kw.pop("verbose", False), show_progress=False)


def _decoder(**kw):
    base, ft = _llm_source_pair(2)
    return DecoderBlockExtender(
        base, ft, infer_family(base), "cpu", verbose=kw.pop("verbose", False), show_progress=False
    )


def _reference_ridge(A, T, lam, ridge_id=0.0):
    A, T = A.float(), T.float()
    mu_A, mu_T = A.mean(dim=0), T.mean(dim=0)
    Ac, Tc = A - mu_A, T - mu_T
    cov = Ac.T @ Ac + (lam + ridge_id) * torch.eye(A.shape[1])
    rhs = Ac.T @ Tc + ridge_id * torch.eye(A.shape[1])
    W_T = torch.linalg.solve(cov, rhs)
    return W_T.T, mu_T - mu_A @ W_T


def _data():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(40, 6, generator=gen), torch.randn(40, 6, generator=gen)


def test_both_extenders_inherit_core():
    assert issubclass(BlockExtender, BlockExtenderCore)
    assert issubclass(DecoderBlockExtender, BlockExtenderCore)
    for name in (
        "_vprint",
        "_fit_ridge",
        "_match_rows",
        "_resolve_depth_delta",
        "_get_ridge",
        "_build_duplication_schedule",
        "_store_input_hook",
        "_store_output_hook",
    ):
        assert name not in vars(BlockExtender) and name not in vars(DecoderBlockExtender), name
        assert getattr(BlockExtender, name) is not None and getattr(DecoderBlockExtender, name) is not None


def test_ridge_fallback_defaults_are_preserved():
    A, T = _data()
    assert DecoderBlockExtender._ridge_weight == 1e-6
    vision, decoder = _vision(), _decoder()
    assert vision._ridge_weight == 1e-6 and decoder._ridge_weight == 1e-6
    ref_W, ref_b = _reference_ridge(A, T, 1e-6)
    for ext in (vision, decoder):
        W, b = ext._fit_ridge(A, T)
        assert torch.allclose(W, ref_W, atol=1e-5) and torch.allclose(b, ref_b, atol=1e-5)
        W2, b2 = ext._fit_ridge(A, T, lambda_reg=1e-6)
        assert torch.equal(W, W2) and torch.equal(b, b2)


def test_vision_ridge_weight_overrides_fallback_but_decoder_never_does():
    A, T = _data()
    vision, decoder = _vision(), _decoder()
    vision._ridge_weight = 0.5
    W, b = vision._fit_ridge(A, T)
    W_ref, b_ref = vision._fit_ridge(A, T, lambda_reg=0.5)
    assert torch.equal(W, W_ref) and torch.equal(b, b_ref)
    assert BlockExtenderCore._ridge_weight == 1e-6 and DecoderBlockExtender._ridge_weight == 1e-6
    W_d, _ = decoder._fit_ridge(A, T)
    assert torch.equal(W_d, decoder._fit_ridge(A, T, lambda_reg=1e-6)[0])
    assert not torch.equal(W, W_d)


def test_ridge_id_and_target_paths():
    A, T = _data()
    target = torch.randn(6, 6, generator=torch.Generator().manual_seed(1))
    for ext in (_vision(), _decoder()):
        W, b = ext._fit_ridge(A, T, lambda_reg=1e-3, ridge_id=0.7, ridge_target=target)
        Ac, Tc = A - A.mean(0), T - T.mean(0)
        cov = Ac.T @ Ac + (1e-3 + 0.7) * torch.eye(6)
        W_T = torch.linalg.solve(cov, Ac.T @ Tc + 0.7 * target)
        assert torch.allclose(W, W_T.T, atol=1e-5)
        assert torch.allclose(b, T.mean(0) - A.mean(0) @ W_T, atol=1e-5)


@pytest.mark.parametrize("make,prefix", [(_vision, "[block_extension] "), (_decoder, "[block_extension_llm] ")])
def test_vprint_prefix_and_verbosity(make, prefix, capsys):
    make(verbose=True)._vprint("hello")
    assert capsys.readouterr().out == f"{prefix}hello\n"
    make(verbose=False)._vprint("hello")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "cls,expected",
    [(BlockExtender, "Expected one of: "), (DecoderBlockExtender, "Expected: ")],
    ids=["vision", "decoder"],
)
def test_duplication_schedule_error_messages_keep_family_wording(cls, expected):
    with pytest.raises(ValueError) as order:
        cls._build_duplication_schedule(4, 2, "sideways", "spread")
    assert str(order.value) == f"Unsupported insertion_order. {expected}bottom-top, top-bottom, random. Got: sideways"
    with pytest.raises(ValueError) as density:
        cls._build_duplication_schedule(4, 2, "bottom-top", "dense")
    assert str(density.value) == f"Unsupported extension_density. {expected}spread, spread_mod, clump. Got: dense"


def test_duplication_schedule_values_are_shared():
    for args in [
        (4, 2, "bottom-top", "spread"),
        (6, 12, "top-bottom", "spread"),
        (5, 3, "bottom-top", "clump"),
        (5, 3, "bottom-top", "spread_mod"),
        (5, 0, "bottom-top", "spread"),
    ]:
        assert BlockExtender._build_duplication_schedule(*args) == DecoderBlockExtender._build_duplication_schedule(
            *args
        )


def test_b18_spread_mod_single_layer_zero_division_in_both_classes():
    for cls in (BlockExtender, DecoderBlockExtender):
        with pytest.raises(ZeroDivisionError):
            cls._build_duplication_schedule(1, 2, "bottom-top", "spread_mod")


def test_match_rows_and_depth_delta():
    A, T = torch.randn(5, 3), torch.randn(7, 3)
    for cls in (BlockExtender, DecoderBlockExtender):
        a, t = cls._match_rows(A, T)
        assert a.shape == t.shape == (5, 3) and torch.equal(t, T[:5])
        assert cls._resolve_depth_delta(4, 2, None) == 2
        assert cls._resolve_depth_delta(4, None, 6) == 2
        assert cls._resolve_depth_delta(4, None, None) == 0
        with pytest.raises(ValueError, match="target_layers_total must be >= 1"):
            cls._resolve_depth_delta(4, None, 0)
        with pytest.raises(ValueError, match="final depth must be >= 1"):
            cls._resolve_depth_delta(4, -4, None)


def test_get_ridge():
    for ext in (_vision(), _decoder()):
        assert ext._get_ridge("q", 0.25) == 0.25
        ext._component_ridge = {"q": 2}
        assert ext._get_ridge("q", 0.25) == 2.0 and isinstance(ext._get_ridge("q", 0.25), float)
        assert ext._get_ridge("k", 0.25) == 0.25


def test_hooks_store_detached_cpu_outputs():
    for cls in (BlockExtender, DecoderBlockExtender):
        store = defaultdict(list)
        lin = torch.nn.Linear(2, 2)
        h1 = lin.register_forward_hook(cls._store_input_hook(store, "in"))
        h2 = lin.register_forward_hook(cls._store_output_hook(store, "out"))
        x = torch.randn(3, 2)
        y = lin(x)
        h1.remove()
        h2.remove()
        assert torch.equal(store["in"][0], x) and torch.equal(store["out"][0], y.detach())
        assert not store["out"][0].requires_grad
