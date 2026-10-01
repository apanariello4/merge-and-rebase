"""Cross-path parity of Ariadne's two activation-storage paths (resident vs streaming).

Both paths fit the same task vector from the same paired calibration batches; they differ only in
HOW the activation statistics are held (full per-batch banks vs Chan-accumulated sufficient
statistics). Contract tested here, on a fixture large enough (width 48, 17 tokens, batch size 8,
8 batches) for the float32 Procrustes factor to differ between the paths in a few entries:

* Task vectors: resident and streaming agree to ~1e-7 relative, not bitwise, because the
  Procrustes cross-covariance is accumulated in a different summation order (measured max relative
  difference 5.6e-7 at width 96). Asserted with ``rtol=1e-5, atol=1e-6`` for the main arm and every
  option supported by BOTH paths, in the extend, shrink and same-depth regimes.
* Diagnostics: the row schema and ``extra`` schema are identical, except for the documented
  resident-only tensors ``q`` (gradient / fidelity_holdout rows) and ``mu_s``/``mu_t``
  (fidelity_holdout rows): internal plumbing for fidelity_holdout that streaming keeps in its
  ``prepared`` dict instead of duplicating d x d tensors into the rows. Those differences are asserted
  EXACTLY, case by case, so any other schema drift fails.
* Gradient Procrustes: the polar factor of a rank-deficient cross-covariance is not unique, so no
  value parity of the task vector is asserted there; instead both paths must flag the rank deficiency
  (``procrustes_q_non_unique``) identically.

Not covered by design: ``block_split`` backfit/joint and the sequential endpoint constructions are
resident-only (streaming rejects them at parse time).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods.ariadne.alignment import (
    apply_depth_pairing_override,
    compute_alignment_diagnostics,
    procrustes_rank_diagnostics,
)
from merge_and_rebase.rebase.methods.ariadne.capture import capture_paired_boundary_activations
from merge_and_rebase.rebase.methods.ariadne.config import parse_direct_residual_config
from merge_and_rebase.rebase.methods.ariadne.method import AriadneRebase

WIDTH, TOKENS, BATCH_SIZE, NUM_BATCHES = 48, 17, 8, 8
N_CLASSES = 6
RTOL, ATOL = 1e-5, 1e-6
DIRECTIONS = {"extend": (2, 4), "shrink": (4, 2), "same": (3, 3)}
OD = ["attn.out_proj", "mlp.c_proj"]

MAIN = dict(
    components=["mlp.c_proj"],
    component_target="block_boundary",
    alignment_map="polar",
    procrustes_source="activation",
    ridge_estimator="empirical_bayes",
    block_split="none",
    num_batches=NUM_BATCHES,
    ridge_relative=0.05,
)

# (case id, config overrides) -- every option supported by BOTH storage paths, one factor at a time.
CASES = [
    ("main", {}),
    ("O+D", dict(components=OD)),
    ("ridge_estimator=fixed_relative", dict(ridge_estimator="fixed_relative")),
    ("ridge_estimator=none", dict(ridge_estimator="none")),
    ("alignment_map=ridge", dict(alignment_map="ridge")),
    ("alignment_map=random_isometry", dict(alignment_map="random_isometry", alignment_seed=3)),
    ("row_weighting=cls_balanced", dict(alignment_row_weighting="cls_balanced")),
    ("row_weighting=delta_magnitude", dict(alignment_row_weighting="delta_magnitude")),
    ("residual_target=transported_endpoint", dict(residual_target="transported_endpoint")),
    ("tv_scaling=global", dict(tv_scaling="global", tv_scaling_iters=2)),
    ("tv_scaling=per_block", dict(tv_scaling="per_block", tv_scaling_iters=2)),
    ("depth_pairing=reversed", dict(depth_pairing="reversed")),
    ("depth_pairing=shift_plus1", dict(depth_pairing="shift_plus1")),
    ("depth_pairing=shift_minus1", dict(depth_pairing="shift_minus1")),
    ("exact_form=False", dict(exact_form=False)),
    ("missing_bias=materialize", dict(missing_bias="materialize")),
    ("realization_diagnostics", dict(realization_diagnostics=True)),
    ("fidelity_holdout", dict(fidelity_holdout=True, fidelity_holdout_batches=3, num_batches=5)),
    ("strength=0.5", dict(strength=0.5)),
    ("strength=2", dict(strength=2.0)),
]
# Resident-only row keys (see module docstring), exactly, per case family.
RESIDENT_ONLY_ROW_KEYS = {"fidelity_holdout": {"q", "mu_s", "mu_t"}}
GRADIENT_RESIDENT_ONLY_ROW_KEYS = {"q"}
# ridge_estimator='none' solves the exact normal equations, which are ill-conditioned (and rejected) for this
# fixture's low-rank GELU features at width 48; it runs on a narrower stack of the same architecture.
FIXTURE_OVERRIDES = {"ridge_estimator=none": dict(width=12)}


@pytest.fixture(autouse=True)
def _single_thread():
    """Tiny matrices: BLAS multithreading only costs time (and CPU on a shared login node)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# --------------------------------------------------------------------------------------
# Fixture: a tiny residual ViT-like stack with the module names the layout expects.
# --------------------------------------------------------------------------------------
class _Attention(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.out_proj = torch.nn.Linear(w, w)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.attn = _Attention(w)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [("c_fc", torch.nn.Linear(w, 2 * w)), ("gelu", torch.nn.GELU()), ("c_proj", torch.nn.Linear(2 * w, w))]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, w, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, w)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(w) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, w, depth):
        super().__init__()
        self.visual = _Visual(w, depth)

    def encode_image(self, x):
        return self.visual(x)


def _tuned(model, seed, scale):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _setup(source_depth, target_depth, *, width=WIDTH, tokens=TOKENS, batch_size=BATCH_SIZE, num_batches=NUM_BATCHES):
    seed = 5
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _tuned(source_base, seed + 1, 0.05)
    generator = torch.Generator().manual_seed(seed + 2)
    n = batch_size * num_batches
    loader = DataLoader(
        TensorDataset(torch.randn(n, tokens, 4, generator=generator), torch.arange(n) % N_CLASSES),
        batch_size=batch_size,
        shuffle=False,
    )
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    return source_base, source_ft, target_base, loader, pairing


def _recipe(width, seed):
    text = torch.randn(N_CLASSES, width, generator=torch.Generator().manual_seed(seed))

    def recipe(model, batch):
        images, labels = batch
        return F.cross_entropy(model.encode_image(images) @ text.T, labels.long()), []

    return recipe


def _prepare(overrides, direction, storage, monkeypatch=None, **fixture_kwargs):
    config = parse_direct_residual_config({**MAIN, **overrides, "activation_storage": storage})
    source_depth, target_depth = DIRECTIONS[direction] if isinstance(direction, str) else direction
    source_base, source_ft, target_base, loader, pairing = _setup(source_depth, target_depth, **fixture_kwargs)
    pairing = apply_depth_pairing_override(pairing, config.depth_pairing)
    kwargs = {}
    if config.procrustes_source == "gradient":
        # The orchestration builds CLIP contrastive recipes from classifiers; this fixture hands recipes through.
        monkeypatch.setattr(
            "merge_and_rebase.models.grad_recipes.clip_contrastive_recipe",
            lambda clf, classnames, build_cfg, device=None, text_features=None: clf,
        )
        kwargs = dict(
            clf_source=_recipe(fixture_kwargs.get("width", WIDTH), 1),
            clf_target=_recipe(fixture_kwargs.get("width", WIDTH), 2),
            classnames=["a"],
            source_build_cfg_task=object(),
            build_cfg_task=object(),
        )
    torch.manual_seed(0)
    return AriadneRebase().prepare(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd={k: v.clone() for k, v in target_base.state_dict().items()},
        source_loader=loader,
        target_loader=loader,
        pairing=pairing,
        config=config,
        device="cpu",
        **kwargs,
    )


def _key_paths(obj, prefix=""):
    """Dict-key paths of a nested structure (list indices collapsed), leaves included."""
    if isinstance(obj, dict):
        out = set()
        for k, v in obj.items():
            out |= _key_paths(v, f"{prefix}/{k}")
        return out or {prefix}
    if isinstance(obj, (list, tuple)):
        out = set()
        for v in obj:
            out |= _key_paths(v, f"{prefix}[]")
        return out or {prefix}
    return {prefix}


def _assert_task_vectors_close(resident, streaming, label):
    for name, a, b in (
        ("unit_task_vector", resident.unit_task_vector, streaming.unit_task_vector),
        ("task_vector", resident.task_vector, streaming.task_vector),
    ):
        assert set(a) == set(b), f"{label}: {name} key sets differ"
        for key in a:
            torch.testing.assert_close(
                a[key], b[key], rtol=RTOL, atol=ATOL, msg=lambda m, k=key, n=name: f"{label} {n}[{k}]: {m}"
            )


def _assert_row_schema_parity(resident, streaming, resident_only, label):
    assert len(resident.diagnostics) == len(streaming.diagnostics), label
    for r_row, s_row in zip(resident.diagnostics, streaming.diagnostics, strict=True):
        assert r_row.get("position") == s_row.get("position") and r_row.get("component") == s_row.get("component")
        assert set(s_row) - set(r_row) == set(), f"{label}: streaming-only row keys {set(s_row) - set(r_row)}"
        assert set(r_row) - set(s_row) == resident_only, f"{label}: resident-only row keys {set(r_row) - set(s_row)}"


def _assert_extra_schema_parity(resident, streaming, label):
    assert set(resident.extra) == set(streaming.extra), label
    for key in resident.extra:
        if key == "alignment_diagnostics" and resident.extra[key] is None:
            assert streaming.extra[key] is None
            continue
        a, b = _key_paths(resident.extra[key], key), _key_paths(streaming.extra[key], key)
        assert a == b, f"{label}: extra[{key!r}] schema differs: only resident {a - b}, only streaming {b - a}"
    assert resident.extra["calibration"] == streaming.extra["calibration"]


# --------------------------------------------------------------------------------------
# Value + schema parity over every option both paths support.
# --------------------------------------------------------------------------------------
_REGIMES = list(DIRECTIONS)
_PARITY_PARAMS = [
    pytest.param(case, overrides, direction, id=f"{case}-{direction}")
    for case, overrides in CASES
    for direction in _REGIMES
]


@pytest.mark.parametrize(("case", "overrides", "direction"), _PARITY_PARAMS)
def test_resident_and_streaming_agree(case, overrides, direction):
    fixture = FIXTURE_OVERRIDES.get(case, {})
    resident = _prepare(overrides, direction, "resident", **fixture)
    streaming = _prepare(overrides, direction, "streaming", **fixture)
    label = f"{case}/{direction}"
    _assert_task_vectors_close(resident, streaming, label)
    _assert_row_schema_parity(resident, streaming, RESIDENT_ONLY_ROW_KEYS.get(case, set()), label)
    _assert_extra_schema_parity(resident, streaming, label)

    # Rank diagnostics are emitted by both paths on every row and agree.
    for r_row, s_row in zip(resident.diagnostics, streaming.diagnostics, strict=True):
        for key in ("procrustes_source", "procrustes_rank", "procrustes_min_dim", "procrustes_q_non_unique"):
            assert r_row[key] == s_row[key], f"{label}: row key {key}"

    # Alignment diagnostics describe the configured map in both paths (values agree up to summation order).
    r_align, s_align = resident.extra["alignment_diagnostics"], streaming.extra["alignment_diagnostics"]
    for j in r_align:
        for key, value in r_align[j].items():
            assert s_align[j][key] == pytest.approx(value, rel=1e-4, abs=1e-8), (
                f"{label}: alignment_diagnostics[{j}][{key}]"
            )


# --------------------------------------------------------------------------------------
# Fix: alignment_diagnostics describe the CONFIGURED map, not a refit uniform polar map.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        dict(alignment_map="ridge"),
        dict(alignment_map="random_isometry", alignment_seed=3),
        dict(alignment_row_weighting="cls_balanced"),
        dict(alignment_row_weighting="delta_magnitude"),
    ],
    ids=["ridge", "random_isometry", "cls_balanced", "delta_magnitude"],
)
def test_resident_alignment_diagnostics_use_configured_map(overrides):
    source_base, source_ft, target_base, loader, pairing = _setup(2, 4)
    captured = capture_paired_boundary_activations(
        source_base, source_ft, target_base, loader, loader, pairing, num_batches=NUM_BATCHES, seed=89, device="cpu"
    )
    uniform_polar = compute_alignment_diagnostics(captured, pairing)
    default_explicit = compute_alignment_diagnostics(
        captured, pairing, alignment_map="polar", alignment_row_weighting="uniform"
    )
    configured = compute_alignment_diagnostics(captured, pairing, **overrides)
    assert default_explicit == uniform_polar  # default path is the historical expression, bit for bit
    assert any(configured[j]["procrustes_error_norm"] != uniform_polar[j]["procrustes_error_norm"] for j in configured)


# --------------------------------------------------------------------------------------
# Gradient Procrustes: rank-deficiency flag in both paths; no value parity (non-unique polar factor).
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", ["extend", "shrink"])
def test_gradient_mode_flags_rank_deficiency_in_both_paths(direction, monkeypatch):
    resident = _prepare(dict(procrustes_source="gradient"), direction, "resident", monkeypatch)
    streaming = _prepare(dict(procrustes_source="gradient"), direction, "streaming", monkeypatch)
    _assert_row_schema_parity(resident, streaming, GRADIENT_RESIDENT_ONLY_ROW_KEYS, f"gradient/{direction}")
    _assert_extra_schema_parity(resident, streaming, f"gradient/{direction}")
    flagged = 0
    for r_row, s_row in zip(resident.diagnostics, streaming.diagnostics, strict=True):
        for row in (r_row, s_row):
            assert row["procrustes_source"] == "gradient"
            assert row["procrustes_q_non_unique"] == (row["procrustes_rank"] < row["procrustes_min_dim"])
        assert r_row["procrustes_rank"] == s_row["procrustes_rank"]
        assert r_row["procrustes_min_dim"] == s_row["procrustes_min_dim"] == WIDTH
        assert r_row["procrustes_q_non_unique"] == s_row["procrustes_q_non_unique"]
        flagged += int(r_row["procrustes_q_non_unique"])
    # A 6-class contrastive loss gives a low-rank gradient cross-covariance at width 48: the flag must fire
    # at (at least) some positions.
    assert flagged > 0
    # activation_gradient_{map_distance,delta_disagreement} are computed in both paths, but their VALUES are not
    # comparable here: they depend on the (non-unique, see above) gradient polar factor.
    for row in (*resident.diagnostics, *streaming.diagnostics):
        for key in ("activation_gradient_map_distance", "activation_gradient_delta_disagreement"):
            assert isinstance(row[key], float) and row[key] == row[key]


def test_activation_rank_deficiency_flagged_and_warned_once(caplog):
    """Fewer calibration rows than the width => rank(cross) < d in the (default polar) activation fit."""
    kwargs = dict(width=WIDTH, tokens=3, batch_size=2, num_batches=1)
    overrides = dict(num_batches=1)
    with caplog.at_level(logging.WARNING, logger="merge_and_rebase.rebase.methods.ariadne.method"):
        resident = _prepare(overrides, "extend", "resident", **kwargs)
        resident_records = [r for r in caplog.records if "rank-deficient" in r.getMessage()]
        caplog.clear()
        streaming = _prepare(overrides, "extend", "streaming", **kwargs)
        streaming_records = [r for r in caplog.records if "rank-deficient" in r.getMessage()]
    for rows, records in ((resident.diagnostics, resident_records), (streaming.diagnostics, streaming_records)):
        assert all(
            row["procrustes_q_non_unique"] and row["procrustes_rank"] < row["procrustes_min_dim"] for row in rows
        )
        assert len(records) == 1  # one aggregated warning, not one per position
        assert str([row["position"] for row in rows]) in records[0].getMessage()
    assert [r["procrustes_rank"] for r in resident.diagnostics] == [r["procrustes_rank"] for r in streaming.diagnostics]


def test_full_rank_activation_fit_is_not_flagged_and_not_warned(caplog):
    with caplog.at_level(logging.WARNING, logger="merge_and_rebase.rebase.methods.ariadne.method"):
        prepared = _prepare({}, "extend", "streaming")
    assert not [r for r in caplog.records if "rank-deficient" in r.getMessage()]
    assert all(
        not row["procrustes_q_non_unique"] and row["procrustes_rank"] == row["procrustes_min_dim"]
        for row in prepared.diagnostics
    )


def test_rank_diagnostics_helper_semantics():
    assert procrustes_rank_diagnostics(3, 5, 4) == {
        "procrustes_rank": 3,
        "procrustes_min_dim": 4,
        "procrustes_q_non_unique": True,
    }
    assert procrustes_rank_diagnostics(4, 5, 4)["procrustes_q_non_unique"] is False
    # Maps that are not the polar factor (ridge, random isometry) report the rank but are never flagged.
    assert procrustes_rank_diagnostics(1, 5, 4, polar_derived=False)["procrustes_q_non_unique"] is False


# --------------------------------------------------------------------------------------
# Calibration metadata must be reproducible across processes (no object ids).
# --------------------------------------------------------------------------------------
def test_calibration_dataset_identity_is_stable():
    a = _prepare({}, "extend", "resident").extra["calibration"]["dataset_identity"]
    b = _prepare({}, "extend", "streaming").extra["calibration"]["dataset_identity"]  # a different dataset object
    assert a == b and "object" not in a and "0x" not in a
    assert a.endswith(f"n={BATCH_SIZE * NUM_BATCHES}")
