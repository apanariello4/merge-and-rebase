"""BRACE block-extension coverage pins that the release golden suite does not have.

``test_release_golden_hashes.py`` pins the default paths of ``BlockExtender`` (vision) and
``DecoderBlockExtender`` (decoder). This module pins every other behaviour axis of the two
classes, one factor at a time from an already-pinned case, so that merging the two classes into
one core with per-family adapters can be checked at hash level:

* decoder: ``duplicate_per_weight``, lmc ``steer``/``shared_ft``, ``skip_correction``,
  ``dampening_factor``, ``component_ridge``, ``n_cascade_iters``, ``share_ft_refs``,
  ``extension_density`` (``spread_mod``/``clump``), ``insertion_order`` (``top-bottom``/``random``),
  plus the config fields the decoder silently ignores (``ridge_weight``, ``collapse_schedule``,
  ``reference_capture``, ``insertion_target_mode``, ``inserted_block_mode``, ``correction_scope``,
  ``target_shared_correction``, ``skip_final_ln``);
* vision: lmc ``steer``, ``reference_capture=eager``, ``inserted_block_mode``, ``correction_scope``,
  ``insertion_target_mode=residual``, ``target_shared_correction``, ``collapse_schedule=disjoint_spans``,
  ``n_cascade_iters``, ``component_ridge``, ``ridge_weight``, ``skip_final_ln``, ``share_ft_refs``,
  density/order, and the diagnostic collector's per-component correction maps;
* plain-data tables for the pure schedule/layout functions of BOTH classes (this is where the
  vision/decoder divergence in ``spread_mod`` collapse and in ``_locate_collapse_pos`` is recorded);
* the validation messages of ``resolve_block_extension_config`` and of the run-time guards.

Hashed cases pin the resulting base and ft ``state_dict`` s, the task vector and (where one is
produced) the layout; each case is executed twice in-process and must agree before the hash is
compared. Hashes and tables were captured at c221d32 + e262b7c-era HEAD (see HASHES.md, "BRACE
extra coverage"); regeneration recipe is in HASHES.md. Never paste a new value over a failing one
without first establishing that the change is an intended, documented numerical change.
"""

from __future__ import annotations

import os
import pprint

import numpy as np
import pytest
import torch

from ._hashing import deterministic_cpu, hash_json
from .test_release_golden_hashes import (
    _LLM_DEPTHS,
    _brace_vision_models,
    _BrModel,
    _BrVisual,
    _class_loader,
    _hash_brace_pair,
    _llm_loader,
    _llm_source_pair,
)

_NP_SEED = 20261001

# --------------------------------------------------------------------------------------
# Expected values: the two literals at the bottom of the file (EXPECTED, TABLES) are
# generated (see HASHES.md, "BRACE extra coverage").
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic():
    with deterministic_cpu(seed=0):
        yield


def _check(name: str, actual: str) -> None:
    capture = os.environ.get("GOLDEN_CAPTURE")
    if capture:  # (re)generation aid only: GOLDEN_CAPTURE=<file> appends "name hash" lines and skips the assert
        with open(capture, "a") as fh:
            fh.write(f"{name} {actual}\n")
        return
    assert name in EXPECTED, f"no expected hash recorded for {name!r} (actual {actual})"
    assert actual == EXPECTED[name], f"{name}: golden hash changed\n  expected {EXPECTED[name]}\n  actual   {actual}"


def _check_table(name: str, actual) -> None:
    if os.environ.get("GOLDEN_CAPTURE"):
        return  # tables are regenerated with ``python -m tests.golden.test_brace_extra_golden``
    assert name in TABLES, f"no expected table recorded for {name!r}"
    assert actual == TABLES[name], f"{name}: table changed"


# --------------------------------------------------------------------------------------
# Config plumbing: every case goes through resolve_block_extension_config, so an invalid
# option combination is rejected exactly as in a real run.
# --------------------------------------------------------------------------------------


def _resolve(params: dict):
    from merge_and_rebase.rebase.block_extension.config import resolve_block_extension_config

    # skip_correction=False: these cases pin the corrected arm (the default is true since 2026-10-09).
    full = {"n_batches_act": 2, "verbose": False, "show_progress": False, "skip_correction": False, **params}
    _, config = resolve_block_extension_config({"block_extension_params": full})
    return config


class _Recorder:
    """Duck-typed diagnostic collector: ``BlockExtender`` only calls ``record_map``."""

    def __init__(self):
        self.records: list[dict] = []

    def record_map(self, *, mode, endpoint, structural_step, final_block, source_block, component, W, b):
        self.records.append(
            {
                "mode": mode,
                "endpoint": endpoint,
                "structural_step": structural_step,
                "final_block": final_block,
                "source_block": source_block,
                "component": component,
                "W": W.detach().clone(),
                "b": b.detach().clone(),
            }
        )


# --------------------------------------------------------------------------------------
# Vision (BlockExtender)
# --------------------------------------------------------------------------------------

_VISION_BASE = {
    "insertion_order": "bottom-top",
    "extension_density": "spread",
    "extension_strategy": "interpolate_per_weight",
    "dampening_factor": 1.0,
    "ridge_identity": 1.0,
    "ridge_weight": 1e-6,
    "lmc_mode": "independent",
    # These cases pin the corrected arm (the default flipped to skip_correction=true on 2026-10-09).
    "skip_correction": False,
}

_VCOMP_RIDGE = {"q": 0.25, "out_proj": 0.5, "c_proj": 2.0}


def _target_model(width=10, depth=5, seed=77):
    """Pretrained-target stand-in for target_shared_correction (wider than the source, deeper)."""
    torch.manual_seed(seed)
    model = _BrModel(depth)
    model.visual = _BrVisual(width=width, depth=depth)
    return model.eval()


def _run_vision(params: dict, source_depth: int, target_depth: int, *, collector=None, with_target=False):
    from merge_and_rebase.rebase.block_extension.vision import run_block_extension

    config = _resolve({**_VISION_BASE, **params})
    base, ft = _brace_vision_models(source_depth)
    loader = _class_loader(n=16, in_dim=6, batch_size=4, seed=3)
    layout: dict = {}
    np.random.seed(_NP_SEED)
    final_depth = run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=loader,
        target_layers_total=target_depth,
        config=config,
        device="cpu",
        diagnostic_collector=collector,
        layout_out=layout,
        target_model=_target_model() if with_target else None,
    )
    assert final_depth == target_depth
    return _hash_brace_pair(base, ft, layout)


# case -> (param overrides on top of _VISION_BASE, source_depth, target_depth)
VISION_CASES = {
    # --- extension 3 -> 5; baseline is the pinned extend_interpolate_independent ---
    "ext_steer": ({"lmc_mode": "steer"}, 3, 5),
    "ext_eager": ({"reference_capture": "eager"}, 3, 5),
    "ext_dup_eager": ({"extension_strategy": "duplicate_per_weight", "reference_capture": "eager"}, 3, 5),
    "ext_share_ft_refs": ({"share_ft_refs": True}, 3, 5),
    "ext_dampening": ({"dampening_factor": 0.5}, 3, 5),
    "ext_dup_dampening": ({"extension_strategy": "duplicate_per_weight", "dampening_factor": 0.5}, 3, 5),
    "ext_skip_correction": ({"skip_correction": True}, 3, 5),
    "ext_identity": ({"skip_correction": True, "inserted_block_mode": "residual_identity"}, 3, 5),
    "ext_identity_inert": ({"skip_correction": True, "inserted_block_mode": "residual_identity_inert"}, 3, 5),
    "ext_scope_interleaved_once": ({"correction_scope": "interleaved_once"}, 3, 5),
    "ext_scope_iterative_all": ({"correction_scope": "iterative_all"}, 3, 5),
    "ext_target_residual": ({"insertion_target_mode": "residual"}, 3, 5),
    "ext_cascade_iters2": ({"n_cascade_iters": 2}, 3, 5),
    "ext_component_ridge": ({"component_ridge": _VCOMP_RIDGE}, 3, 5),
    "ext_ridge_weight": ({"ridge_weight": 1e-2}, 3, 5),
    "ext_skip_final_ln": ({"skip_final_ln": True}, 3, 5),
    "ext_order_top_bottom": ({"insertion_order": "top-bottom"}, 3, 5),
    "ext_order_random": ({"insertion_order": "random"}, 3, 5),
    "ext_density_spread_mod": ({"extension_density": "spread_mod"}, 3, 5),
    "ext_density_clump": ({"extension_density": "clump"}, 3, 5),
    # --- extension 4 -> 6 (order/density select different blocks here) ---
    "ext4_spread": ({}, 4, 6),
    "ext4_order_top_bottom": ({"insertion_order": "top-bottom"}, 4, 6),
    "ext4_order_random": ({"insertion_order": "random"}, 4, 6),
    "ext4_density_spread_mod": ({"extension_density": "spread_mod"}, 4, 6),
    "ext4_density_clump": ({"extension_density": "clump"}, 4, 6),
    # --- reduction 3 -> 2; baseline is the pinned shrink_interpolate_independent ---
    "shr_steer": ({"lmc_mode": "steer"}, 3, 2),
    "shr_eager": ({"reference_capture": "eager"}, 3, 2),
    "shr_share_ft_refs": ({"share_ft_refs": True}, 3, 2),
    "shr_dampening": ({"dampening_factor": 0.5}, 3, 2),
    "shr_dup": ({"extension_strategy": "duplicate_per_weight"}, 3, 2),
    "shr_skip_correction": ({"skip_correction": True}, 3, 2),
    "shr_cascade_iters2": ({"n_cascade_iters": 2}, 3, 2),
    "shr_component_ridge": ({"component_ridge": _VCOMP_RIDGE}, 3, 2),
    "shr_ridge_weight": ({"ridge_weight": 1e-2}, 3, 2),
    "shr_skip_final_ln": ({"skip_final_ln": True}, 3, 2),
    # --- reduction 6 -> 4 (schedules differ by order/density here); baseline shr6_spread ---
    "shr6_spread": ({}, 6, 4),
    "shr6_order_top_bottom": ({"insertion_order": "top-bottom"}, 6, 4),
    "shr6_order_random": ({"insertion_order": "random"}, 6, 4),
    "shr6_density_spread_mod": ({"extension_density": "spread_mod"}, 6, 4),
    "shr6_density_clump": ({"extension_density": "clump"}, 6, 4),
    "shr6_clump_top_bottom": ({"extension_density": "clump", "insertion_order": "top-bottom"}, 6, 4),
    "shr6_clump_random": ({"extension_density": "clump", "insertion_order": "random"}, 6, 4),
    "shr6_disjoint": ({"collapse_schedule": "disjoint_spans"}, 6, 4),
    "shr6_disjoint_top_bottom": ({"collapse_schedule": "disjoint_spans", "insertion_order": "top-bottom"}, 6, 4),
    "shr6_disjoint_steer": ({"collapse_schedule": "disjoint_spans", "lmc_mode": "steer"}, 6, 4),
    "shr4_disjoint": ({"collapse_schedule": "disjoint_spans"}, 4, 2),
}


@pytest.mark.parametrize("case", sorted(VISION_CASES))
def test_brace_vision_extra(case):
    params, source_depth, target_depth = VISION_CASES[case]
    first = _run_vision(params, source_depth, target_depth)
    second = _run_vision(params, source_depth, target_depth)
    assert first == second, f"{case}: not deterministic across two in-process runs"
    for part, digest in first.items():
        _check(f"brace_x_vision:{case}:{part}", digest)


def test_brace_vision_target_shared_correction():
    """Target-informed blend of the shared c_proj target (needs the pretrained target backbone)."""
    params = {"lmc_mode": "shared", "target_shared_correction": {"target_weight": 0.5}}
    first = _run_vision(params, 3, 5, with_target=True)
    second = _run_vision(params, 3, 5, with_target=True)
    assert first == second
    for part, digest in first.items():
        _check(f"brace_x_vision:ext_target_shared_correction:{part}", digest)


def _diag_digest(records):
    return hash_json(records)


@pytest.mark.parametrize("lmc_mode", ["independent", "steer", "shared", "shared_ft"])
def test_brace_vision_diagnostic_collector_shrink(lmc_mode):
    """Per-component correction maps seen by the diagnostic collector (reduction path)."""
    runs = []
    for _ in range(2):
        collector = _Recorder()
        parts = _run_vision({"lmc_mode": lmc_mode}, 3, 2, collector=collector)
        assert collector.records, "reduction path must record correction maps"
        runs.append((parts, _diag_digest(collector.records), len(collector.records)))
    assert runs[0] == runs[1]
    parts, diag, n_records = runs[0]
    for part, digest in parts.items():
        _check(f"brace_x_vision:shr_diag_{lmc_mode}:{part}", digest)
    _check(f"brace_x_vision:shr_diag_{lmc_mode}:diag", diag)
    assert n_records > 0


def test_brace_vision_diagnostic_collector_extend_records_nothing():
    """Surprise pinned on purpose: the extension path never sets ``_diagnostic_context``.

    ``_record_correction`` is a no-op while the context is ``None``; only ``_shrink_per_weight``
    assigns it. The collector therefore stays empty for every extension, in every lmc mode.
    """
    for lmc_mode in ("independent", "steer", "shared", "shared_ft"):
        collector = _Recorder()
        _run_vision({"lmc_mode": lmc_mode}, 3, 5, collector=collector)
        assert collector.records == [], lmc_mode


def test_brace_vision_collector_does_not_change_state():
    """Attaching a collector is observation only: model hashes equal the collector-free run."""
    with_collector = _run_vision({"lmc_mode": "steer"}, 3, 2, collector=_Recorder())
    without = _run_vision({"lmc_mode": "steer"}, 3, 2)
    assert with_collector == without


def test_brace_vision_eager_equals_lazy():
    """Documented relation between the two capture schedules (informational pin)."""
    lazy = _run_vision({}, 3, 5)
    eager = _run_vision({"reference_capture": "eager"}, 3, 5)
    _check("brace_x_vision:eager_equals_lazy_ext", hash_json({"equal": lazy == eager}))
    lazy = _run_vision({}, 3, 2)
    eager = _run_vision({"reference_capture": "eager"}, 3, 2)
    _check("brace_x_vision:eager_equals_lazy_shr", hash_json({"equal": lazy == eager}))


# --------------------------------------------------------------------------------------
# Decoder (DecoderBlockExtender), real tiny Qwen2
# --------------------------------------------------------------------------------------

_DCOMP_RIDGE = {"q_proj": 0.25, "o_proj": 0.5, "down_proj": 2.0}

# Pins the corrected arm (the default flipped to skip_correction=true on 2026-10-09); cases may override it.
_DECODER_BASE = {"extension_strategy": "interpolate_per_weight", "lmc_mode": "independent", "skip_correction": False}


def _run_decoder(params: dict, source_depth: int, target_depth: int):
    from merge_and_rebase.rebase.block_extension.decoder import run_block_extension_llm
    from merge_and_rebase.rebase.model_families import infer_family

    config = _resolve({**_DECODER_BASE, **params})
    base, ft = _llm_source_pair(source_depth)
    layout: dict = {}
    np.random.seed(_NP_SEED)
    final_depth = run_block_extension_llm(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_llm_loader(),
        target_layers_total=target_depth,
        config=config,
        family_adapter=infer_family(base),
        device="cpu",
        layout_out=layout,
    )
    assert final_depth == target_depth
    return _hash_brace_pair(base, ft, layout)


# case -> param overrides on top of _DECODER_BASE (baseline: interpolate_per_weight, independent,
# bottom-top, spread, ridge_identity 0, which is the pinned brace_decoder_independent).
DECODER_CASES = {
    "dup": {"extension_strategy": "duplicate_per_weight"},
    "dup_shared": {"extension_strategy": "duplicate_per_weight", "lmc_mode": "shared"},
    "steer": {"lmc_mode": "steer"},
    # steer only differs from independent through the ridge pull toward the base map (ridge_identity > 0)
    "steer_ridge1": {"lmc_mode": "steer", "ridge_identity": 1.0},
    "shared_ft": {"lmc_mode": "shared_ft"},
    "skip_correction": {"skip_correction": True},
    "dampening": {"dampening_factor": 0.5},
    "dup_dampening": {"extension_strategy": "duplicate_per_weight", "dampening_factor": 0.5},
    "cascade_iters2": {"n_cascade_iters": 2},
    "component_ridge": {"component_ridge": _DCOMP_RIDGE},
    "ridge_identity": {"ridge_identity": 1.0},
    "share_ft_refs": {"share_ft_refs": True},
    "order_top_bottom": {"insertion_order": "top-bottom"},
    "order_random": {"insertion_order": "random"},
    "density_spread_mod": {"extension_density": "spread_mod"},
    "density_clump": {"extension_density": "clump"},
}

# Deeper pair (4 -> 6 / 6 -> 4) where the schedule options actually select different blocks.
DECODER_DEEP_CASES = {
    "deep_spread": {},
    "deep_order_top_bottom": {"insertion_order": "top-bottom"},
    "deep_order_random": {"insertion_order": "random"},
    "deep_density_spread_mod": {"extension_density": "spread_mod"},
    "deep_density_clump": {"extension_density": "clump"},
    "deep_clump_top_bottom": {"extension_density": "clump", "insertion_order": "top-bottom"},
    "deep_clump_random": {"extension_density": "clump", "insertion_order": "random"},
}

_DEEP_DEPTHS = {"extend": (4, 6), "shrink": (6, 4)}


def _decoder_digests(params, depths):
    first = _run_decoder(params, *depths)
    second = _run_decoder(params, *depths)
    assert first == second, "decoder case not deterministic across two in-process runs"
    return first


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("case", sorted(DECODER_CASES))
def test_brace_decoder_extra(case, direction):
    for part, digest in _decoder_digests(DECODER_CASES[case], _LLM_DEPTHS[direction]).items():
        _check(f"brace_x_decoder:{case}:{direction}:{part}", digest)


# Known decoder crash, pinned as a crash: clump + top-bottom shrink anchors every step at the same
# original block; once that block sits in the last span, the decoder's `_locate_collapse_pos` returns
# the last position and `chain[pos + 1]` raises IndexError (vision clamps and succeeds).
_DECODER_DEEP_SHRINK_CRASHES = {"deep_clump_top_bottom"}


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("case", sorted(DECODER_DEEP_CASES))
def test_brace_decoder_deep(case, direction):
    if direction == "shrink" and case in _DECODER_DEEP_SHRINK_CRASHES:
        with pytest.raises(IndexError):
            _run_decoder(DECODER_DEEP_CASES[case], *_DEEP_DEPTHS[direction])
        return
    for part, digest in _decoder_digests(DECODER_DEEP_CASES[case], _DEEP_DEPTHS[direction]).items():
        _check(f"brace_x_decoder:{case}:{direction}:{part}", digest)


# Options carried by BlockExtensionConfig that DecoderBlockExtender never reads. Each one must leave
# every hash identical to the baseline run (all of them resolve without complaint, so a user can
# believe they are active). A decoder adapter that starts honouring one of them changes results.
DECODER_IGNORED = {
    "ridge_weight": ({"ridge_weight": 1e-2}, {}),
    "collapse_schedule_disjoint": ({"collapse_schedule": "disjoint_spans"}, {}),
    "reference_capture_eager": ({"reference_capture": "eager"}, {}),
    "insertion_target_mode_residual": ({"insertion_target_mode": "residual"}, {}),
    "skip_final_ln": ({"skip_final_ln": True}, {}),
    "correction_scope_interleaved_once": ({"correction_scope": "interleaved_once"}, {}),
    "inserted_block_mode_identity": (
        {"skip_correction": True, "inserted_block_mode": "residual_identity"},
        {"skip_correction": True},
    ),
    "target_shared_correction": (
        {"lmc_mode": "shared", "target_shared_correction": {"target_weight": 0.5}},
        {"lmc_mode": "shared"},
    ),
}


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("case", sorted(DECODER_IGNORED))
def test_brace_decoder_ignores_vision_only_options(case, direction):
    params, baseline = DECODER_IGNORED[case]
    depths = _LLM_DEPTHS[direction]
    assert _run_decoder(params, *depths) == _run_decoder(baseline, *depths)


def test_brace_decoder_dup_vs_interpolate():
    """duplicate_per_weight differs from interpolate on extension but is IDENTICAL on shrink.

    Surprise pinned on purpose: the decoder's `_shrink_per_weight` calls `_interpolate_block_weights`
    unconditionally, ignoring `per_weight_mode`, whereas vision skips it for `duplicate_per_weight`
    (see the vision `shr_dup` pin, which differs from `shr_steer`'s interpolate baseline).
    """
    dup = {"extension_strategy": "duplicate_per_weight"}
    assert _run_decoder(dup, *_LLM_DEPTHS["extend"]) != _run_decoder({}, *_LLM_DEPTHS["extend"])
    assert _run_decoder(dup, *_LLM_DEPTHS["shrink"]) == _run_decoder({}, *_LLM_DEPTHS["shrink"])


# --------------------------------------------------------------------------------------
# Validation messages
# --------------------------------------------------------------------------------------

_RESOLVE_ERRORS = [
    ({"correction_scope": "iterative_all", "skip_correction": True}, "requires skip_correction=false"),
    (
        {"correction_scope": "interleaved_once", "inserted_block_mode": "residual_identity"},
        "requires inserted_block_mode='ariadne'",
    ),
    ({"correction_scope": "bogus"}, "correction_scope must be"),
    ({"inserted_block_mode": "residual_identity"}, "requires skip_correction=true"),
    ({"inserted_block_mode": "bogus"}, "inserted_block_mode must be"),
    ({"transport_activation_mode": "interpolate_neighbors"}, "requires skip_correction=true"),
    ({"reference_capture": "bogus"}, "reference_capture must be 'lazy' or 'eager'"),
    ({"target_shared_correction": {"target_weight": 0.5}}, "requires lmc_mode='shared'"),
    (
        {"target_shared_correction": {"target_weight": 0.5}, "lmc_mode": "shared", "skip_correction": True},
        "requires skip_correction=false",
    ),
    (
        {"target_shared_correction": {"target_weight": 0.5, "num_batches": 3}, "lmc_mode": "shared"},
        "must equal n_batches_act=2",
    ),
    ({"target_shared_correction": {"target_weight": -1.0}}, "target_weight must be >= 0"),
    ({"n_batches_act": 0}, "n_batches_act must be > 0"),
    ({"ridge_weight": -1.0}, "ridge_weight must be >= 0"),
    ({"ridge_identity": -1.0}, "ridge_identity must be >= 0"),
]


@pytest.mark.parametrize(("params", "message"), _RESOLVE_ERRORS, ids=[m for _, m in _RESOLVE_ERRORS])
def test_resolve_block_extension_config_rejects(params, message):
    with pytest.raises(ValueError, match=message):
        _resolve(params)


def test_resolve_accepts_every_pinned_vision_case():
    """Every option set pinned above is a valid config (shrink-only rejections happen at run time)."""
    for params, _, _ in VISION_CASES.values():
        _resolve({**_VISION_BASE, **params})


_VISION_SHRINK_ERRORS = [
    ({"skip_correction": True, "inserted_block_mode": "residual_identity"}, "is an extension baseline"),
    ({"correction_scope": "interleaved_once"}, "is an extension option; block shrink has no insertion"),
    (
        {"lmc_mode": "shared", "target_shared_correction": {"target_weight": 0.5}},
        "target_shared_correction is an extension option",
    ),
    ({"collapse_schedule": "bogus"}, "Unsupported collapse_schedule 'bogus'"),
    (
        {"collapse_schedule": "disjoint_spans", "insertion_order": "random"},
        "'random' has no meaning for a fixed disjoint partition",
    ),
]


@pytest.mark.parametrize(("params", "message"), _VISION_SHRINK_ERRORS, ids=[m[:40] for _, m in _VISION_SHRINK_ERRORS])
def test_vision_shrink_rejects_extension_only_options(params, message):
    with pytest.raises(ValueError, match=message):
        _run_vision(params, 3, 2)


_RUNTIME_ERRORS = [
    ({"insertion_order": "bogus"}, "Unsupported insertion_order"),
    ({"extension_density": "bogus"}, "Unsupported extension_density"),
    ({"lmc_mode": "bogus"}, "Unsupported lmc_mode 'bogus'"),
    ({"extension_strategy": "interpolate"}, "Vision extension_strategy='interpolate' is no longer supported"),
    ({"extension_strategy": "bogus"}, "Unsupported vision extension_strategy"),
]


@pytest.mark.parametrize(("params", "message"), _RUNTIME_ERRORS, ids=[m for _, m in _RUNTIME_ERRORS])
def test_vision_extension_runtime_errors(params, message):
    with pytest.raises(ValueError, match=message):
        _run_vision(params, 3, 5)


_DECODER_RUNTIME_ERRORS = [
    ({"insertion_order": "bogus"}, "Unsupported insertion_order"),
    ({"extension_density": "bogus"}, "Unsupported extension_density"),
    ({"lmc_mode": "bogus"}, "Unsupported lmc_mode 'bogus'"),
    ({"extension_strategy": "interpolate"}, "extension_strategy 'interpolate' \\(non per-weight\\) is disabled"),
    ({"extension_strategy": "bogus"}, "Unsupported extension_strategy 'bogus'"),
]


@pytest.mark.parametrize(
    ("params", "message"), _DECODER_RUNTIME_ERRORS, ids=[m[:40] for _, m in _DECODER_RUNTIME_ERRORS]
)
def test_decoder_extension_runtime_errors(params, message):
    with pytest.raises(ValueError, match=message):
        _run_decoder(params, *_LLM_DEPTHS["extend"])


# --------------------------------------------------------------------------------------
# Pure schedule / layout tables (plain data, no hashing)
# --------------------------------------------------------------------------------------

_ORDERS = ("bottom-top", "top-bottom", "random")
_DENSITIES = ("spread", "spread_mod", "clump")
# (curr_layers, n_blocks_to_add): n < curr, n == curr and n > curr (wrap-around) all appear.
_DUP_PAIRS = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 5), (4, 2), (4, 4), (4, 6), (6, 3), (6, 8), (8, 3)]
# (curr_layers, n_blocks_to_remove): n_to_remove <= curr - 1, except curr=1 which must raise.
_COLLAPSE_PAIRS = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (6, 2), (6, 3), (6, 5), (8, 3), (8, 4)]
_CHAINS = {
    "singletons4": [(0,), (1,), (2,), (3,)],
    "head_merged": [(0, 1), (2,), (3,)],
    "mid_merged": [(0,), (1, 2), (3,)],
    "tail_merged": [(0,), (1,), (2, 3)],
    "all_but_last": [(0, 1, 2), (3,)],
}


def _capture(fn, *args):
    """Seeded call: the result as a list, or ``'ERR <ExceptionType>'`` (messages are pinned separately)."""
    np.random.seed(_NP_SEED)
    try:
        return list(fn(*args))
    except Exception as exc:  # noqa: BLE001 - recording the failure mode is the point
        return f"ERR {type(exc).__name__}"


def _extenders():
    from merge_and_rebase.rebase.block_extension.decoder import DecoderBlockExtender
    from merge_and_rebase.rebase.block_extension.vision import BlockExtender

    return {"vision": BlockExtender, "decoder": DecoderBlockExtender}


def _dup_schedule_tables():
    out = {}
    for family, cls in _extenders().items():
        table = {}
        for curr, n in _DUP_PAIRS:
            for order in _ORDERS:
                for density in _DENSITIES:
                    table[f"{curr},{n},{order},{density}"] = _capture(
                        cls._build_duplication_schedule, curr, n, order, density
                    )
        out[f"dup_schedule:{family}"] = table
    return out


def _collapse_schedule_tables():
    out = {}
    for family, cls in _extenders().items():
        table = {}
        for curr, n in _COLLAPSE_PAIRS:
            for order in _ORDERS:
                for density in _DENSITIES:
                    table[f"{curr},{n},{order},{density}"] = _capture(
                        cls._build_collapse_schedule, curr, n, order, density
                    )
        out[f"collapse_schedule:{family}"] = table
    return out


def _simulate_collapse(cls, curr, schedule):
    """Replay the shrink loop's span bookkeeping with the class's own ``_locate_collapse_pos``."""
    chain = [{"orig_idxs": (i,)} for i in range(curr)]
    for anchor in schedule:
        pos = cls._locate_collapse_pos(chain, anchor)
        merged = tuple(chain[pos]["orig_idxs"] + chain[pos + 1]["orig_idxs"])
        chain[pos : pos + 2] = [{"orig_idxs": merged}]
    return [item["orig_idxs"] for item in chain]


def _realized_span_tables():
    out = {}
    for family, cls in _extenders().items():
        table = {}
        for curr, n in _COLLAPSE_PAIRS:
            for order in _ORDERS:
                for density in _DENSITIES:
                    np.random.seed(_NP_SEED)
                    try:
                        schedule = cls._build_collapse_schedule(curr, n, order, density)
                        table[f"{curr},{n},{order},{density}"] = _simulate_collapse(cls, curr, schedule)
                    except Exception as exc:  # noqa: BLE001
                        table[f"{curr},{n},{order},{density}"] = f"ERR {type(exc).__name__}"
        out[f"realized_spans:{family}"] = table
    return out


def _locate_tables():
    out = {}
    for family, cls in _extenders().items():
        table = {}
        for name, spans in _CHAINS.items():
            chain = [{"orig_idxs": span} for span in spans]
            for anchor in range(-1, spans[-1][-1] + 2):
                try:
                    table[f"{name},{anchor}"] = cls._locate_collapse_pos(chain, anchor)
                except Exception as exc:  # noqa: BLE001
                    table[f"{name},{anchor}"] = f"ERR {type(exc).__name__}"
        out[f"locate_collapse_pos:{family}"] = table
    return out


def _pure_function_tables():
    from merge_and_rebase.rebase.block_extension.schedules import (
        balanced_collapse_spans,
        build_extension_layout,
        build_reduction_layout,
        disjoint_collapse_schedule,
        plan_inserted_positions,
        spread_anchor_schedule,
    )

    out = {}
    out["spread_anchor_schedule"] = {
        f"{k},{p},{order}": _capture(spread_anchor_schedule, k, p, order)
        for k in (0, 1, 2, 3, 5, 7)
        for p in (0, 1, 2, 3, 5, 8)
        for order in (*_ORDERS, "bogus")
    }
    spans_pairs = [(1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (5, 3), (6, 4), (7, 3), (8, 3), (3, 4), (0, 1)]
    out["balanced_collapse_spans"] = {
        f"{curr},{final},{order}": _capture(balanced_collapse_spans, curr, final, order)
        for curr, final in spans_pairs
        for order in _ORDERS
    }
    out["disjoint_collapse_schedule"] = {
        f"{curr},{n},{order}": _capture(disjoint_collapse_schedule, curr, n, order)
        for curr, n in _COLLAPSE_PAIRS
        for order in _ORDERS
    }
    plan = {}
    for curr, n in _DUP_PAIRS:
        for order in ("bottom-top", "top-bottom"):
            for density in _DENSITIES:
                np.random.seed(_NP_SEED)
                key = f"{curr},{n},{order},{density}"
                try:
                    schedule = _extenders()["vision"]._build_duplication_schedule(curr, n, order, density)
                    plan[key] = {"schedule": list(schedule), "positions": plan_inserted_positions(curr, schedule)}
                except Exception as exc:  # noqa: BLE001
                    plan[key] = f"ERR {type(exc).__name__}"
    out["plan_inserted_positions"] = plan

    def extension_chain(curr, schedule):
        chain = [{"orig_idx": i, "inserted": False} for i in range(curr)]
        for src in schedule:
            pos = max(i for i, item in enumerate(chain) if item["orig_idx"] == src) + 1
            chain.insert(pos, {"orig_idx": src, "inserted": True, "neighbour_orig_idx": min(src + 1, curr - 1)})
        return chain

    out["build_extension_layout"] = {
        f"{curr}:{'+'.join(map(str, schedule))}": build_extension_layout(extension_chain(curr, schedule))
        for curr, schedule in ((3, [0, 1]), (3, [2]), (4, [0, 0, 3]), (2, [0, 1, 1]))
    }
    out["build_reduction_layout"] = {
        name: build_reduction_layout([{"orig_idxs": span} for span in spans]) for name, spans in _CHAINS.items()
    }
    return out


def _all_tables():
    tables = {}
    tables.update(_dup_schedule_tables())
    tables.update(_collapse_schedule_tables())
    tables.update(_realized_span_tables())
    tables.update(_locate_tables())
    tables.update(_pure_function_tables())
    return tables


_PURE_TABLE_NAMES = [
    "dup_schedule:vision",
    "dup_schedule:decoder",
    "collapse_schedule:vision",
    "collapse_schedule:decoder",
    "realized_spans:vision",
    "realized_spans:decoder",
    "locate_collapse_pos:vision",
    "locate_collapse_pos:decoder",
    "spread_anchor_schedule",
    "balanced_collapse_spans",
    "disjoint_collapse_schedule",
    "plan_inserted_positions",
    "build_extension_layout",
    "build_reduction_layout",
]


@pytest.mark.parametrize("name", _PURE_TABLE_NAMES)
def test_brace_pure_tables(name):
    _check_table(name, _all_tables_cached()[name])


_TABLE_CACHE: dict = {}


def _all_tables_cached():
    if not _TABLE_CACHE:
        first = _all_tables()
        assert first == _all_tables(), "schedule tables are not deterministic under a fixed np.random seed"
        _TABLE_CACHE.update(first)
    return _TABLE_CACHE


def test_schedule_divergences_are_pinned_as_such():
    """The facts the tables record, stated explicitly (a Phase-6 core must keep both policies)."""
    tables = _all_tables_cached()
    # spread_mod collapse: vision spreads the anchors with np.linspace, the decoder reuses the
    # duplication rule `i % (curr - 1)`.
    assert tables["collapse_schedule:vision"]["8,3,bottom-top,spread_mod"] == [0, 3, 6]
    assert tables["collapse_schedule:decoder"]["8,3,bottom-top,spread_mod"] == [0, 1, 2]
    # _locate_collapse_pos: vision clamps the anchor inside the last span to len(chain) - 2, the
    # decoder returns len(chain) - 1 (and the following ``chain[pos + 1]`` raises IndexError).
    assert tables["locate_collapse_pos:vision"]["singletons4,3"] == 2
    assert tables["locate_collapse_pos:decoder"]["singletons4,3"] == 3
    assert tables["collapse_schedule:vision"]["4,2,top-bottom,spread_mod"] == [2, 0]
    assert tables["collapse_schedule:decoder"]["4,2,top-bottom,spread_mod"] == [0, 1]
    assert tables["realized_spans:vision"]["4,2,top-bottom,spread_mod"] == [(0, 1), (2, 3)]
    assert tables["realized_spans:decoder"]["4,2,top-bottom,spread_mod"] == [(0, 1, 2), (3,)]
    # clump + top-bottom collapse repeats the same anchor: fine for vision, IndexError for the decoder.
    assert tables["realized_spans:vision"]["4,2,top-bottom,clump"] == [(0,), (1, 2, 3)]
    assert tables["realized_spans:decoder"]["4,2,top-bottom,clump"] == "ERR IndexError"


_SCHEDULE_ERROR_MESSAGES = [
    ("vision", "_build_duplication_schedule", (3, 2, "bogus", "spread"), "Unsupported insertion_order"),
    ("decoder", "_build_duplication_schedule", (3, 2, "bogus", "spread"), "Unsupported insertion_order"),
    ("vision", "_build_duplication_schedule", (3, 2, "bottom-top", "bogus"), "Unsupported extension_density"),
    ("decoder", "_build_duplication_schedule", (3, 2, "bottom-top", "bogus"), "Unsupported extension_density"),
    ("vision", "_build_collapse_schedule", (1, 1, "bottom-top", "spread"), "depth is less than 2"),
    ("decoder", "_build_collapse_schedule", (1, 1, "bottom-top", "spread"), "depth is less than 2"),
    ("vision", "_build_collapse_schedule", (4, 2, "bogus", "clump"), "Unsupported insertion_order"),
    ("vision", "_build_collapse_schedule", (4, 2, "bogus", "spread_mod"), "Unsupported insertion_order"),
    (
        "vision",
        "_locate_collapse_pos",
        ([{"orig_idxs": (0,)}, {"orig_idxs": (1,)}], 5),
        "Could not locate collapse anchor 5",
    ),
    (
        "decoder",
        "_locate_collapse_pos",
        ([{"orig_idxs": (0,)}, {"orig_idxs": (1,)}], 5),
        "Could not locate anchor_orig_idx=5",
    ),
]


@pytest.mark.parametrize(
    ("family", "fn_name", "args", "message"),
    _SCHEDULE_ERROR_MESSAGES,
    ids=[f"{f}-{n}-{i}" for i, (f, n, _, _) in enumerate(_SCHEDULE_ERROR_MESSAGES)],
)
def test_schedule_error_messages(family, fn_name, args, message):
    with pytest.raises((ValueError, ZeroDivisionError), match=message):
        getattr(_extenders()[family], fn_name)(*args)


def test_vision_spread_mod_duplication_single_block_divides_by_zero():
    """curr_layers=1 with spread_mod hits ``i % 0`` in both classes (pinned, not fixed)."""
    for cls in _extenders().values():
        with pytest.raises(ZeroDivisionError):
            cls._build_duplication_schedule(1, 2, "bottom-top", "spread_mod")


# --------------------------------------------------------------------------------------
# GENERATED LITERALS. Regenerate: see HASHES.md ("BRACE extra coverage").
# --------------------------------------------------------------------------------------
EXPECTED: dict[str, str] = {
    "brace_x_decoder:cascade_iters2:extend:base_state": "8b349351634b73bd932687befc8ae59982f798b8919306847e6d2992c5a96115",
    "brace_x_decoder:cascade_iters2:extend:ft_state": "6f98f67878364444670545758edc06e1608aac84192f50bd4964f49b30a2e91b",
    "brace_x_decoder:cascade_iters2:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:cascade_iters2:extend:task_vector": "6ac8e267940c9379a376cfe89aac8274d9e2a3bbd6b6ab40a8ea9b2aef861d86",
    "brace_x_decoder:cascade_iters2:shrink:base_state": "f13a3a5abfc090114c42bf5c20bea8959bc2a107a319a08e500161b35dae54a4",
    "brace_x_decoder:cascade_iters2:shrink:ft_state": "6968735e37ffec4e504d8a5fc38ab2c280fda70d7fa2713949d75ebc50da2535",
    "brace_x_decoder:cascade_iters2:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:cascade_iters2:shrink:task_vector": "d112ae19baa394bea395609e11769d758a8c064b8cb7202118976a392717b5d1",
    "brace_x_decoder:component_ridge:extend:base_state": "32fa861a46d5dcca253f8dd36c64d11167b7b53592aa6981b31bbd67aeb67e75",
    "brace_x_decoder:component_ridge:extend:ft_state": "c787beeb7f037250e19c7d26e5a51562ef3149a12ebf8eade310dea666e36e65",
    "brace_x_decoder:component_ridge:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:component_ridge:extend:task_vector": "349aa1b39c8ad8ea3ae3d3529614cfd74dff5c4c281f33ace6425df24df2663a",
    "brace_x_decoder:component_ridge:shrink:base_state": "8971e54d46571d5a3ae325a66a418a6bdcc5f8a8353bf0a3ff1a6ceee18ac0c3",
    "brace_x_decoder:component_ridge:shrink:ft_state": "61124d8ebff80a16b8597f7fa3fb89fe787d8f5b9242f22a97e94e49671680f2",
    "brace_x_decoder:component_ridge:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:component_ridge:shrink:task_vector": "e6f39ea8a8d9fd8ece4a06c06972e6d6be34ea6e168ccb6777b24c2923a18aed",
    "brace_x_decoder:dampening:extend:base_state": "4406d19dd51872883687638031def74f85f095feb7c06fb014e906f325105036",
    "brace_x_decoder:dampening:extend:ft_state": "fd521b59f978021ba27bca653ae52e2e5832577f10f34354cc2d61ff725d9f6c",
    "brace_x_decoder:dampening:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:dampening:extend:task_vector": "a741729477b98e6357933f385b61724c7cbc04f2972016518e9087d2fe32fb3b",
    "brace_x_decoder:dampening:shrink:base_state": "2e4f719b966b75e61542cdfcec206242e2fbfabee0c95852a66490093af55b1f",
    "brace_x_decoder:dampening:shrink:ft_state": "a4acadb34add7b391724cded91ce9a2190dd98d7ef36b4ac8bb01bb34fa4e420",
    "brace_x_decoder:dampening:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:dampening:shrink:task_vector": "399741fd141de947cce4817b95c50e7b3856fab4baa82e5806c70df21f9a024d",
    "brace_x_decoder:deep_clump_random:extend:base_state": "23551038c4f3b28b334b08c7f9a5374809c5ba377dcaddfab8c798806f10531d",
    "brace_x_decoder:deep_clump_random:extend:ft_state": "f7cda90a7be639c09696291e7dbb9b1a996fcd0cb02569b40e21eda38aa4fdc5",
    "brace_x_decoder:deep_clump_random:extend:layout": "52c20bbf6e9398923e34f1022f6d9a58759d766ac484127c3c87e7194e6c54cd",
    "brace_x_decoder:deep_clump_random:extend:task_vector": "54f9b8732e0237a120f3268b4b72bb0084808ac938ff169a8327bde0f7413a03",
    "brace_x_decoder:deep_clump_random:shrink:base_state": "c0707cec9c904ac416163b219025208bfe9c295dcef55cfccd0d73b157477089",
    "brace_x_decoder:deep_clump_random:shrink:ft_state": "de608f6b0e68072c857e684c826e45a4feb701a24758988f87242d9ddaff62a6",
    "brace_x_decoder:deep_clump_random:shrink:layout": "50610d7a4a4eebed762a74b77e4c32a46d04f740ad958247a2e7bf37154f214e",
    "brace_x_decoder:deep_clump_random:shrink:task_vector": "0d05dc8e929605bfe16ffb43921e167e1dc103d77dd2e1248cc8b545c016e4d6",
    "brace_x_decoder:deep_clump_top_bottom:extend:base_state": "4ae6d2fcd9b8fd21aa3c67bf969746c01a71821053e7adfaf81d81cddebcf984",
    "brace_x_decoder:deep_clump_top_bottom:extend:ft_state": "58029f1752486ce2b9aaea3eeaff55db8d0de0e6b89b7c49daf1fe093d021e51",
    "brace_x_decoder:deep_clump_top_bottom:extend:layout": "46790e464092a288ffbe35d50e87eaacbee86e8afca14bac2eeec7645df81e38",
    "brace_x_decoder:deep_clump_top_bottom:extend:task_vector": "999e32b0ae7c5649fd9892fc0e472deda25502c33f0ed3390bbc91363fb42911",
    "brace_x_decoder:deep_density_clump:extend:base_state": "23551038c4f3b28b334b08c7f9a5374809c5ba377dcaddfab8c798806f10531d",
    "brace_x_decoder:deep_density_clump:extend:ft_state": "f7cda90a7be639c09696291e7dbb9b1a996fcd0cb02569b40e21eda38aa4fdc5",
    "brace_x_decoder:deep_density_clump:extend:layout": "52c20bbf6e9398923e34f1022f6d9a58759d766ac484127c3c87e7194e6c54cd",
    "brace_x_decoder:deep_density_clump:extend:task_vector": "54f9b8732e0237a120f3268b4b72bb0084808ac938ff169a8327bde0f7413a03",
    "brace_x_decoder:deep_density_clump:shrink:base_state": "4cc53f2945992625513298527ce99ee05026793fdf7a3bbe2f09cf515d7732a6",
    "brace_x_decoder:deep_density_clump:shrink:ft_state": "a30c35aaf036e4659d99a0189fd446aa537595ee70fbfedf30785fcf0c033fa5",
    "brace_x_decoder:deep_density_clump:shrink:layout": "50610d7a4a4eebed762a74b77e4c32a46d04f740ad958247a2e7bf37154f214e",
    "brace_x_decoder:deep_density_clump:shrink:task_vector": "ed9fc23ab74be991ec9697f6f685fb35ab6f28320fcc4d64eda2b06138409aef",
    "brace_x_decoder:deep_density_spread_mod:extend:base_state": "82e39450626e53a6b45563f12c4b3c7c3250d1c6961722954f3f26d0df2ff198",
    "brace_x_decoder:deep_density_spread_mod:extend:ft_state": "27d86105430f69c59d0545393348c322d60acd39c9080c00e3298b94ceeeb646",
    "brace_x_decoder:deep_density_spread_mod:extend:layout": "eb5a33f000546d4367ac2c93db38cc313d9fb4b00b16fc17f4ec6851369c2b85",
    "brace_x_decoder:deep_density_spread_mod:extend:task_vector": "e22afe1acdd63518f982b2bbdb5cbdb1a2cec461929cd5b5b234e759a324dfad",
    "brace_x_decoder:deep_density_spread_mod:shrink:base_state": "4cc53f2945992625513298527ce99ee05026793fdf7a3bbe2f09cf515d7732a6",
    "brace_x_decoder:deep_density_spread_mod:shrink:ft_state": "a30c35aaf036e4659d99a0189fd446aa537595ee70fbfedf30785fcf0c033fa5",
    "brace_x_decoder:deep_density_spread_mod:shrink:layout": "50610d7a4a4eebed762a74b77e4c32a46d04f740ad958247a2e7bf37154f214e",
    "brace_x_decoder:deep_density_spread_mod:shrink:task_vector": "ed9fc23ab74be991ec9697f6f685fb35ab6f28320fcc4d64eda2b06138409aef",
    "brace_x_decoder:deep_order_random:extend:base_state": "ccf15dad0a12c1f3b58c1580c10047aeb58bf5ebd5e3a16dd52ee1539b0bf585",
    "brace_x_decoder:deep_order_random:extend:ft_state": "058c402b86bf6c1ad5372adddf7ec677b9bc4daee0c114973c6db4fd80c34781",
    "brace_x_decoder:deep_order_random:extend:layout": "e517136ec1b0d11f7890dc8cc7de33120026768b1ab76689fd8e152e7a3de579",
    "brace_x_decoder:deep_order_random:extend:task_vector": "649fe729f6ea0a9449464f6e66509e59a596e2236078fbc72c40f45c94aefe54",
    "brace_x_decoder:deep_order_random:shrink:base_state": "f63a9c3e48a5b145f7525c272abc6b0835c74c0f9e442352ce579ee7c34b97a6",
    "brace_x_decoder:deep_order_random:shrink:ft_state": "b6fa02bae118fe7c58ab25436e3b709ea9b6db1ddd00bfc5977177d9c914284b",
    "brace_x_decoder:deep_order_random:shrink:layout": "1c0695b0ca16441d4728a7d222041f14b2a97ed318799da384ea1ba05565c2b3",
    "brace_x_decoder:deep_order_random:shrink:task_vector": "b7e972fe3c755057c6d793448118c1473aa8e4cb8a2d914a77d29b075fb54b40",
    "brace_x_decoder:deep_order_top_bottom:extend:base_state": "5358bc1380e240f3195d22e53feded100c745d00b4aefda62162279b5c3b9223",
    "brace_x_decoder:deep_order_top_bottom:extend:ft_state": "aa6390945b1aa24c24ae3e53af15ffe417bd9a30af7410c195a3c73a8802bb28",
    "brace_x_decoder:deep_order_top_bottom:extend:layout": "e517136ec1b0d11f7890dc8cc7de33120026768b1ab76689fd8e152e7a3de579",
    "brace_x_decoder:deep_order_top_bottom:extend:task_vector": "525d2de2bddba019d2a10363a0fa5d10bf7fb9b4c583d80f467b395fc1577628",
    "brace_x_decoder:deep_order_top_bottom:shrink:base_state": "c9c7cb42da325df0032c79756298a2c153d468d39b4f6c156d985e3f1bc2435f",
    "brace_x_decoder:deep_order_top_bottom:shrink:ft_state": "c8980016e811ebbc3cdfde440377c7a853182c5124c959f57609e9d0bdf97ee7",
    "brace_x_decoder:deep_order_top_bottom:shrink:layout": "bc60a3c802358f82e7bc4ed72d3ed875ec87fdc7b44907d37ee20f8dbd22853e",
    "brace_x_decoder:deep_order_top_bottom:shrink:task_vector": "5d930e8c8fa68b45b692796a172878a86d100cdbc2611f4cc857d08518583c46",
    "brace_x_decoder:deep_spread:extend:base_state": "82e39450626e53a6b45563f12c4b3c7c3250d1c6961722954f3f26d0df2ff198",
    "brace_x_decoder:deep_spread:extend:ft_state": "27d86105430f69c59d0545393348c322d60acd39c9080c00e3298b94ceeeb646",
    "brace_x_decoder:deep_spread:extend:layout": "eb5a33f000546d4367ac2c93db38cc313d9fb4b00b16fc17f4ec6851369c2b85",
    "brace_x_decoder:deep_spread:extend:task_vector": "e22afe1acdd63518f982b2bbdb5cbdb1a2cec461929cd5b5b234e759a324dfad",
    "brace_x_decoder:deep_spread:shrink:base_state": "b6b98d022921475a4d874ec9522c4d4ed740db1a1ce0172187fff4794b68332f",
    "brace_x_decoder:deep_spread:shrink:ft_state": "6afb5383437010ab525d024bdb2281a96af9ee4b0c08f15e24f5fffa73a38fa1",
    "brace_x_decoder:deep_spread:shrink:layout": "1b1fb0ab68216cf9c323efc849bc144c0f940ea75fd4f85e84771758501597ab",
    "brace_x_decoder:deep_spread:shrink:task_vector": "504b97dd24c24a4213cd1b1a9b1b11361adc810b7e5e97857a917c12f8e99def",
    "brace_x_decoder:density_clump:extend:base_state": "a04b5e2d3d8d45803faf24200ba5d086b23f2caa10c1b6c48d3fe8f953dd4364",
    "brace_x_decoder:density_clump:extend:ft_state": "3fce549edaca720d42ef8629a1f8109060e2325fd47f0bd3d720b03c2286d303",
    "brace_x_decoder:density_clump:extend:layout": "6a57d3adf563b75cd13234b710ed7a4ff0a85130befeefe8e48fbea80b161dee",
    "brace_x_decoder:density_clump:extend:task_vector": "0aaa86b6cc9d6852d4fe6d8550052182c1b484c471e5f75af034fe335c23cb53",
    "brace_x_decoder:density_clump:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:density_clump:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:density_clump:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:density_clump:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_x_decoder:density_spread_mod:extend:base_state": "a04b5e2d3d8d45803faf24200ba5d086b23f2caa10c1b6c48d3fe8f953dd4364",
    "brace_x_decoder:density_spread_mod:extend:ft_state": "3fce549edaca720d42ef8629a1f8109060e2325fd47f0bd3d720b03c2286d303",
    "brace_x_decoder:density_spread_mod:extend:layout": "6a57d3adf563b75cd13234b710ed7a4ff0a85130befeefe8e48fbea80b161dee",
    "brace_x_decoder:density_spread_mod:extend:task_vector": "0aaa86b6cc9d6852d4fe6d8550052182c1b484c471e5f75af034fe335c23cb53",
    "brace_x_decoder:density_spread_mod:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:density_spread_mod:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:density_spread_mod:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:density_spread_mod:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_x_decoder:dup:extend:base_state": "e663a6004774506227920b0ffd851bdf2d1df2cb99dc415787e6fe20d20d5a48",
    "brace_x_decoder:dup:extend:ft_state": "f397df894f9c41dc6fa784aaef34da8d51015f9e6004b539110cb4d77dba7ec4",
    "brace_x_decoder:dup:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:dup:extend:task_vector": "cd2f97789213603fc6a888d73b21710349b87a628c28a688b48992062802d192",
    "brace_x_decoder:dup:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:dup:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:dup:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:dup:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_x_decoder:dup_dampening:extend:base_state": "2e07f7de39c5710c6f4ed7329bd4a55f52c8f39ed84322c07e40c433360c892e",
    "brace_x_decoder:dup_dampening:extend:ft_state": "e75db128f85e57954f67c8157a8fd09d62581fcfd189b5a5387c30bcb01cc06b",
    "brace_x_decoder:dup_dampening:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:dup_dampening:extend:task_vector": "259efb6648d7d768887676334090782350520014a85217e1883ccf2d04eb1e0a",
    "brace_x_decoder:dup_dampening:shrink:base_state": "2e4f719b966b75e61542cdfcec206242e2fbfabee0c95852a66490093af55b1f",
    "brace_x_decoder:dup_dampening:shrink:ft_state": "a4acadb34add7b391724cded91ce9a2190dd98d7ef36b4ac8bb01bb34fa4e420",
    "brace_x_decoder:dup_dampening:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:dup_dampening:shrink:task_vector": "399741fd141de947cce4817b95c50e7b3856fab4baa82e5806c70df21f9a024d",
    "brace_x_decoder:dup_shared:extend:base_state": "e663a6004774506227920b0ffd851bdf2d1df2cb99dc415787e6fe20d20d5a48",
    "brace_x_decoder:dup_shared:extend:ft_state": "15b48a0274ff2588aba54ee690fa6b3af5dbfe949a93e53556a779253d785660",
    "brace_x_decoder:dup_shared:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:dup_shared:extend:task_vector": "762a28e66ffdc776b22f074f5a983b2f3a0d6311ba03e871e39545d42e3f785f",
    "brace_x_decoder:dup_shared:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:dup_shared:shrink:ft_state": "82409d5fd7c4cbe5eacb89ab07dad9cb20c95ace83447cacbed012c59b59730b",
    "brace_x_decoder:dup_shared:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:dup_shared:shrink:task_vector": "dcfdd8db68ee8378c0c4d522dbc5946cc61227a93115ca26e5f102e7f93d1b0d",
    "brace_x_decoder:order_random:extend:base_state": "5368f19e8d519e049dfd30008e201632707b1662e3abbd4447b1e16759ba5402",
    "brace_x_decoder:order_random:extend:ft_state": "7950bca2d2b67a453cfd353b4385a60e560ff85de36c11cc19300af043a723ed",
    "brace_x_decoder:order_random:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:order_random:extend:task_vector": "ea820242f4468ca4fea130fcea9836c7f41c8a967d11aa4909f43e42b738a82b",
    "brace_x_decoder:order_random:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:order_random:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:order_random:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:order_random:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_x_decoder:order_top_bottom:extend:base_state": "5368f19e8d519e049dfd30008e201632707b1662e3abbd4447b1e16759ba5402",
    "brace_x_decoder:order_top_bottom:extend:ft_state": "7950bca2d2b67a453cfd353b4385a60e560ff85de36c11cc19300af043a723ed",
    "brace_x_decoder:order_top_bottom:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:order_top_bottom:extend:task_vector": "ea820242f4468ca4fea130fcea9836c7f41c8a967d11aa4909f43e42b738a82b",
    "brace_x_decoder:order_top_bottom:shrink:base_state": "042f4c59fec2a492d363da9cad5660c465c840ffb220a953a343517088130cbe",
    "brace_x_decoder:order_top_bottom:shrink:ft_state": "3052296765bb7bd5718179035610ee278081a96cf144a8742c822db7204f5dd3",
    "brace_x_decoder:order_top_bottom:shrink:layout": "43258c0308781c19435f3145ab227088e25bb59676a6d740914444fd2088e421",
    "brace_x_decoder:order_top_bottom:shrink:task_vector": "e35c33012bce285ee058df52257b510bd4965612bf0068389496c364665085d4",
    "brace_x_decoder:ridge_identity:extend:base_state": "a26edcdd00d41f4774d77dde859f8711ddb75019936832da91976e2092e28b1c",
    "brace_x_decoder:ridge_identity:extend:ft_state": "4144a2493cf2bef3040381c2424e119790ea88dfd68f56ebba09b51de186f62b",
    "brace_x_decoder:ridge_identity:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:ridge_identity:extend:task_vector": "8946737a96aaaf7395d3ef9133ed8f1a62c72f516b0e4d5e13a551d35b97154b",
    "brace_x_decoder:ridge_identity:shrink:base_state": "921d28b71392a165b9ca68b90b7292106d3ac9b5402d87cdcf5a6cd458232a5a",
    "brace_x_decoder:ridge_identity:shrink:ft_state": "33960f690c76d5210bda44743709549e466bb5a282173393d024cf4a3ace3ba1",
    "brace_x_decoder:ridge_identity:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:ridge_identity:shrink:task_vector": "716572c4915213fea33b1a6c650252cea1a6a263fc7a296cc3be446d991506f8",
    "brace_x_decoder:share_ft_refs:extend:base_state": "0c56ee7e97b5e491b10c5dce691e049782998bc99a7d11d6e97ac4687c9f3606",
    "brace_x_decoder:share_ft_refs:extend:ft_state": "f69a0d35d9478fc0aa9fffb62657623f151c877c90c04d1d78e9aafb825e432f",
    "brace_x_decoder:share_ft_refs:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:share_ft_refs:extend:task_vector": "5dbd826ca1783f7e7adee90da929621ee910a174bbfc98c713582d6f12302fb0",
    "brace_x_decoder:share_ft_refs:shrink:base_state": "4fe510ecfce621c7bcf5d5dd4adcb410958496e46164e5a3c125dbbc176d6a8d",
    "brace_x_decoder:share_ft_refs:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:share_ft_refs:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:share_ft_refs:shrink:task_vector": "bb0272429c0bdd085ba1fa0329364bfc7031e621617075c8ab685c42bf9e29aa",
    "brace_x_decoder:shared_ft:extend:base_state": "6c665c42889f5af63535b4da7c26e0f6ac36d42944a14f7c868f483e268c8f77",
    "brace_x_decoder:shared_ft:extend:ft_state": "f69a0d35d9478fc0aa9fffb62657623f151c877c90c04d1d78e9aafb825e432f",
    "brace_x_decoder:shared_ft:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:shared_ft:extend:task_vector": "6aed59c42037117c5c4a26e3b17ee07bb8bab477db6a436fd0926aaf55cbc526",
    "brace_x_decoder:shared_ft:shrink:base_state": "e22c7a640c59858b78d10fb00d4660d3de6f0092ca4a19987798351ea4773b0e",
    "brace_x_decoder:shared_ft:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:shared_ft:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:shared_ft:shrink:task_vector": "9583b88ee173f30ea3dce4a351fa94594b02fd63557ea936fd0ac248ac448bb4",
    "brace_x_decoder:skip_correction:extend:base_state": "2dd663b10c1b0a7db84df89230e2f7d64d0041ca0ecca6cdd1bcef18dd2042c5",
    "brace_x_decoder:skip_correction:extend:ft_state": "a90ccbf57f2b02fc6fe8d74ea8f863d45931d6f1e022cc98d2aa6b468d72b377",
    "brace_x_decoder:skip_correction:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:skip_correction:extend:task_vector": "48abebcfa2030d800cdff4c20663c1205ac45d961f5bd56f798928bdc33cfc9a",
    "brace_x_decoder:skip_correction:shrink:base_state": "e3999cc4e2c4bdfe12a8110b30377444090ade9f97ab5dd66b8434b2f6afecfe",
    "brace_x_decoder:skip_correction:shrink:ft_state": "f820f2ffe9fffce92131178d091f2b9f941a314e290b3d13055105f1d6ea48d0",
    "brace_x_decoder:skip_correction:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:skip_correction:shrink:task_vector": "d5844ab6e5db0eddbf2679f1108466c52cadfadf36dbc551deeeeaf40dc6fc31",
    "brace_x_decoder:steer:extend:base_state": "e1aed95cfd7b20c7491faf29512838fe3943e56ea1a75e42847f3715d67165a3",
    "brace_x_decoder:steer:extend:ft_state": "f69a0d35d9478fc0aa9fffb62657623f151c877c90c04d1d78e9aafb825e432f",
    "brace_x_decoder:steer:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:steer:extend:task_vector": "9a38e272c443a77474fcbd4226c994696641bbe521ecd5937c0400ebaba6f9f2",
    "brace_x_decoder:steer:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_x_decoder:steer:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_x_decoder:steer:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:steer:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_x_decoder:steer_ridge1:extend:base_state": "a26edcdd00d41f4774d77dde859f8711ddb75019936832da91976e2092e28b1c",
    "brace_x_decoder:steer_ridge1:extend:ft_state": "3d532d33389680493922c8e0fde0e3b67a7d090923fa83611ad09f7409ea82f9",
    "brace_x_decoder:steer_ridge1:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_x_decoder:steer_ridge1:extend:task_vector": "ca9e8749ce3b339072f86b52bd36e447aea3cd4152bf6633d1b03dc7001bb0d0",
    "brace_x_decoder:steer_ridge1:shrink:base_state": "921d28b71392a165b9ca68b90b7292106d3ac9b5402d87cdcf5a6cd458232a5a",
    "brace_x_decoder:steer_ridge1:shrink:ft_state": "f40b0c0f55f280cebd891ce63524cf9b40ce3291f8725481d17e37f22f2505cc",
    "brace_x_decoder:steer_ridge1:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_x_decoder:steer_ridge1:shrink:task_vector": "6a0beb6c8319c8ddde2ed86ecd331aaaa9828a7661242f0ebcdffe8e459555c8",
    "brace_x_vision:eager_equals_lazy_ext": "fc925fdc26787f3e33affd87ca47beb70ab4b133f00ef1849049eff64132bf53",
    "brace_x_vision:eager_equals_lazy_shr": "fc925fdc26787f3e33affd87ca47beb70ab4b133f00ef1849049eff64132bf53",
    "brace_x_vision:ext4_density_clump:base_state": "6dbdee11d8f92568f1fdac5f1fd55cce33f575f12e0d5aee93099941ebdba953",
    "brace_x_vision:ext4_density_clump:ft_state": "a4df9e363e517eceb17f20859c1b83c7884030f9079cfe76102557672c1499fb",
    "brace_x_vision:ext4_density_clump:layout": "52c20bbf6e9398923e34f1022f6d9a58759d766ac484127c3c87e7194e6c54cd",
    "brace_x_vision:ext4_density_clump:task_vector": "23b765c103832137f0a1c8b70359a639a5899867b5e0196a4243bc81274f6b3d",
    "brace_x_vision:ext4_density_spread_mod:base_state": "6d14f33bc360650d1d46e7d019866c825afbdf25fc480f0acffba3b0f4cdcf94",
    "brace_x_vision:ext4_density_spread_mod:ft_state": "2a2c046699fbc7d7ec32f4212fae7e5bdc8d035629a6da0c59c55f90401df944",
    "brace_x_vision:ext4_density_spread_mod:layout": "eb5a33f000546d4367ac2c93db38cc313d9fb4b00b16fc17f4ec6851369c2b85",
    "brace_x_vision:ext4_density_spread_mod:task_vector": "eeafcc548161ac0cef723c54c2664063b5647826d6378e6fa023a39fb8bc20ff",
    "brace_x_vision:ext4_order_random:base_state": "a5cc16aeb961e1590bcd2fe02da4b889c875b793174fb1b4f14efb1c4254b4a5",
    "brace_x_vision:ext4_order_random:ft_state": "a4f7d5c6d099a200bbe084bcb64ad5ab71eeaa180fb6c6f3fd7ecff9124a2a16",
    "brace_x_vision:ext4_order_random:layout": "e517136ec1b0d11f7890dc8cc7de33120026768b1ab76689fd8e152e7a3de579",
    "brace_x_vision:ext4_order_random:task_vector": "5c72380272e1f4473460e100e5bd879630133440b87aba9e381c64bea5b0aeb5",
    "brace_x_vision:ext4_order_top_bottom:base_state": "78e2f735ff6c00d2bfbb8b861fe11733af623c0a57d2d5d61b7e5ef44efdc7b3",
    "brace_x_vision:ext4_order_top_bottom:ft_state": "bfbb07c8c940f2f232f55224799b3f298967375d749ad6699947f8ff1f871c08",
    "brace_x_vision:ext4_order_top_bottom:layout": "e517136ec1b0d11f7890dc8cc7de33120026768b1ab76689fd8e152e7a3de579",
    "brace_x_vision:ext4_order_top_bottom:task_vector": "37094d18c8f0ae868e493dc60f62556153f62ba813186413f41d7c398991691b",
    "brace_x_vision:ext4_spread:base_state": "6d14f33bc360650d1d46e7d019866c825afbdf25fc480f0acffba3b0f4cdcf94",
    "brace_x_vision:ext4_spread:ft_state": "2a2c046699fbc7d7ec32f4212fae7e5bdc8d035629a6da0c59c55f90401df944",
    "brace_x_vision:ext4_spread:layout": "eb5a33f000546d4367ac2c93db38cc313d9fb4b00b16fc17f4ec6851369c2b85",
    "brace_x_vision:ext4_spread:task_vector": "eeafcc548161ac0cef723c54c2664063b5647826d6378e6fa023a39fb8bc20ff",
    "brace_x_vision:ext_cascade_iters2:base_state": "08598d2a50c6a15d53abc4c4096888ec18c7b57b3a503b5f959ed2466a08e6c5",
    "brace_x_vision:ext_cascade_iters2:ft_state": "479ef63ed884ead9b06991659ea25d0ad8ae5ee5be3e9f7b67ae81e2fa122e3b",
    "brace_x_vision:ext_cascade_iters2:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_cascade_iters2:task_vector": "3a644fcb24f5dd7d37d72f992685be52461ecf696e6bc2abc6b2f4608c97f34c",
    "brace_x_vision:ext_component_ridge:base_state": "31afa2a18c769bb0e5ce09cc68a99ca033579423019e0e931607a06acfe0be8f",
    "brace_x_vision:ext_component_ridge:ft_state": "338acbbddf6345722cc53e19db42d6a3de032ccc1e3ab79d330ff92a16ed70ac",
    "brace_x_vision:ext_component_ridge:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_component_ridge:task_vector": "f5c9b9da31bc801910fe1fee1ac75e5f060e882fc9019d2971b564237159797c",
    "brace_x_vision:ext_dampening:base_state": "e7c14354ddec412cfb28707c92f0aff9c23d276e28ea7a2d0d5cdc00e8e7f2fc",
    "brace_x_vision:ext_dampening:ft_state": "276dd7afeb2c3560171a072a3fc95de9ddf7f456402485b86eb093125263bc9a",
    "brace_x_vision:ext_dampening:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_dampening:task_vector": "a87b09b91cac6a9472ab16bcacbf075a7c5fb261793db9934c9b0cd370b4d4d4",
    "brace_x_vision:ext_density_clump:base_state": "4db72e706320e046611054f7c4a37b5a4d5afebf22574c5ab0795b97cf65c239",
    "brace_x_vision:ext_density_clump:ft_state": "9788cb6a1d92ce9d2db49e24bef4a4b195c245422eafbe12ca85c570698054db",
    "brace_x_vision:ext_density_clump:layout": "f5d76a571530a7ff7ae5b6ebb3f2227573b5fb8da086649fe78c78fbb07b9a5d",
    "brace_x_vision:ext_density_clump:task_vector": "24e8e9464871063b82c3829dc5d631ecb5422e81fb0a69fc842e20c720a39758",
    "brace_x_vision:ext_density_spread_mod:base_state": "d2927777ed73705995f8662f0285bb21da478e8d582455a784e1d33ee5a5a7f3",
    "brace_x_vision:ext_density_spread_mod:ft_state": "6ae5584333dc8265eec8ad14d47710c784f8e5edfdd7162c3dfa67a08482a42f",
    "brace_x_vision:ext_density_spread_mod:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_density_spread_mod:task_vector": "18b42ed4476040e02df930f6ce29d36fadf40a6317523fba2cdc8c7206b540f8",
    "brace_x_vision:ext_dup_dampening:base_state": "47a3dfe3c06f1c6a5ee61808f81e95e62e24b23f277f8e44a470fd91fe08fa99",
    "brace_x_vision:ext_dup_dampening:ft_state": "7c4535a95f717fb4837ef7bb203df8fe818a2247d6b363b3e2145e9a1c310410",
    "brace_x_vision:ext_dup_dampening:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_dup_dampening:task_vector": "5c2cc25348337ca1b80425bd6eedf997890621a0f7dcccd6eccce64352cb4078",
    "brace_x_vision:ext_dup_eager:base_state": "4a3edd070afa6742fea1da2c07cc5bc5d1cb2756990b3416973a0a027b1a4e88",
    "brace_x_vision:ext_dup_eager:ft_state": "54c7348ece68a3e95d663e29cde1097925cee15881ee0c4b70847357d5cfb1fa",
    "brace_x_vision:ext_dup_eager:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_dup_eager:task_vector": "e4c6e223f68f77bbbc42d841194b7cd676104a1ec77c3fc562c4a7cf369c3e93",
    "brace_x_vision:ext_eager:base_state": "d2927777ed73705995f8662f0285bb21da478e8d582455a784e1d33ee5a5a7f3",
    "brace_x_vision:ext_eager:ft_state": "6ae5584333dc8265eec8ad14d47710c784f8e5edfdd7162c3dfa67a08482a42f",
    "brace_x_vision:ext_eager:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_eager:task_vector": "18b42ed4476040e02df930f6ce29d36fadf40a6317523fba2cdc8c7206b540f8",
    "brace_x_vision:ext_identity:base_state": "45b202a38296b65f0f6ddbba60ab1e310bb6dc06a9d040743c2cb7ccd6e29b59",
    "brace_x_vision:ext_identity:ft_state": "8d96785a3d94a255827b4635f12e5c2dc078367be3b60c801dab2bdddf39122f",
    "brace_x_vision:ext_identity:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_identity:task_vector": "bc8e82f58693e931da8af6ce9df6e5d54ac66437b1db0a22a68762ce67b67c57",
    "brace_x_vision:ext_identity_inert:base_state": "45b202a38296b65f0f6ddbba60ab1e310bb6dc06a9d040743c2cb7ccd6e29b59",
    "brace_x_vision:ext_identity_inert:ft_state": "8d03a6328427a1bc400bf899ceaee62ca4d473fa29f040570307321b49bf7c7a",
    "brace_x_vision:ext_identity_inert:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_identity_inert:task_vector": "c4183d435a4ac5913dc3aebe2d0f4e894701c3f65d8d2b55292789d072f2be44",
    "brace_x_vision:ext_order_random:base_state": "b32389f8e2a68a47116c52d396ac91cac839edee238f3289ff2e3119fae7d438",
    "brace_x_vision:ext_order_random:ft_state": "9aaa3b5438b5d04164172cd4734d7aabe70a2c86b9bb7ed9e6fc245a6ca8b3e5",
    "brace_x_vision:ext_order_random:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_order_random:task_vector": "fcaf4e07554de003f3f31964dd15a6f55890145fd59e440f50f633c4d56e3893",
    "brace_x_vision:ext_order_top_bottom:base_state": "b32389f8e2a68a47116c52d396ac91cac839edee238f3289ff2e3119fae7d438",
    "brace_x_vision:ext_order_top_bottom:ft_state": "9aaa3b5438b5d04164172cd4734d7aabe70a2c86b9bb7ed9e6fc245a6ca8b3e5",
    "brace_x_vision:ext_order_top_bottom:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_order_top_bottom:task_vector": "fcaf4e07554de003f3f31964dd15a6f55890145fd59e440f50f633c4d56e3893",
    "brace_x_vision:ext_ridge_weight:base_state": "0e7f0be73c6082ac22d491ae98999fa39e7922825c7c26a27a6707b70330788a",
    "brace_x_vision:ext_ridge_weight:ft_state": "f7c4f588522901203ca56701f0702fcb7f7a6eaceadc65e8b39d09a9d1f3cac9",
    "brace_x_vision:ext_ridge_weight:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_ridge_weight:task_vector": "e33c788d07616e25c5c8c1cc3d5683f46ff05aa211703292577c0aa09a8a1eb2",
    "brace_x_vision:ext_scope_interleaved_once:base_state": "539ecec3597d500462319296f4d7b3c6d5ee397f3fb1a40f38c4f66aceca7bb8",
    "brace_x_vision:ext_scope_interleaved_once:ft_state": "5ff9721da5c6aaf92ca01fa8ec40ffef648d10028d13bc9300dee60c9e91f8fd",
    "brace_x_vision:ext_scope_interleaved_once:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_scope_interleaved_once:task_vector": "167a000e042248bd762121dd86326837e3bb0db6bf36c62b1a802d197cb3b471",
    "brace_x_vision:ext_scope_iterative_all:base_state": "2884f38cc90d5c159dd6e1b9a198bf67e2c555dd8cbd99ba55dfceb9275a8ba7",
    "brace_x_vision:ext_scope_iterative_all:ft_state": "e0e5d219556e0290125a81c6bec8bbf80ae5caa3a9af481c3e35656aeb1541e1",
    "brace_x_vision:ext_scope_iterative_all:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_scope_iterative_all:task_vector": "e25b331ad48efbbd290647507cae42f2cd9fac3b5d79bba95ee2451dcc5ca14d",
    "brace_x_vision:ext_share_ft_refs:base_state": "06910b64996a931e217e442e87772db314d70a38ba6a959c77d911e18c3c0989",
    "brace_x_vision:ext_share_ft_refs:ft_state": "6ae5584333dc8265eec8ad14d47710c784f8e5edfdd7162c3dfa67a08482a42f",
    "brace_x_vision:ext_share_ft_refs:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_share_ft_refs:task_vector": "fc9f657b798977b1246c886c57e2ec0faf7b2e45005212f1b3297b92e86e3f57",
    "brace_x_vision:ext_skip_correction:base_state": "edbecb9241830069b6fa861d13385769cb3a638d26bbeb603fd845f88014503c",
    "brace_x_vision:ext_skip_correction:ft_state": "c84de3fcd7598ee137aae625a8251ad9f89ed4addcd93db643d6eaadd49bceec",
    "brace_x_vision:ext_skip_correction:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_skip_correction:task_vector": "764938e6c05a33436228a32f543e985b9ac555e0e1abdad4ba7132de9244fada",
    "brace_x_vision:ext_skip_final_ln:base_state": "d2927777ed73705995f8662f0285bb21da478e8d582455a784e1d33ee5a5a7f3",
    "brace_x_vision:ext_skip_final_ln:ft_state": "6ae5584333dc8265eec8ad14d47710c784f8e5edfdd7162c3dfa67a08482a42f",
    "brace_x_vision:ext_skip_final_ln:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_skip_final_ln:task_vector": "18b42ed4476040e02df930f6ce29d36fadf40a6317523fba2cdc8c7206b540f8",
    "brace_x_vision:ext_steer:base_state": "d2927777ed73705995f8662f0285bb21da478e8d582455a784e1d33ee5a5a7f3",
    "brace_x_vision:ext_steer:ft_state": "425a101a1d2b84a7a4c9cee25d10810e245dcf8c71e5e7b898246ad4c41912ab",
    "brace_x_vision:ext_steer:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_steer:task_vector": "2dad56b4d7aa8749b161ca6554597964a21111dd54b573924eb109bb985b1dfa",
    "brace_x_vision:ext_target_residual:base_state": "df38e2124e7c228ad45d33b3a46d25961d8e6e6f7fca914c733773055ee95a01",
    "brace_x_vision:ext_target_residual:ft_state": "e05c5b9a08e610ed61e8031de9e5227d0fc4873145238b9b110b3bd3034a9213",
    "brace_x_vision:ext_target_residual:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_target_residual:task_vector": "92967e9421d0d42bd5caf8d8de1c8e9dda843af82bff88ccf9d5319e53fb6eca",
    "brace_x_vision:ext_target_shared_correction:base_state": "538f55b3d9b84ef66d3110f5975c326b62c89693ed2bccab130e8982433f90e6",
    "brace_x_vision:ext_target_shared_correction:ft_state": "f5f890caef939a7dd6fdaf6e3cf24474e3d6009bdb4353091a8568bf67962637",
    "brace_x_vision:ext_target_shared_correction:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_x_vision:ext_target_shared_correction:task_vector": "5c7e56487e6f92480ebf179a79ce6b0b40260885f32e7e17379099be3a8fa738",
    "brace_x_vision:shr4_disjoint:base_state": "ac458a3cf3c93e01538c5ca20b3b3ce541dad2ae826b420bf946879463edaeab",
    "brace_x_vision:shr4_disjoint:ft_state": "89e2d46ea25ed2eb834c11b7ad4e4539b95194b5e50f0e5b6aa0fbc2e445c545",
    "brace_x_vision:shr4_disjoint:layout": "1ef00da49b5428febfc7232d862b7018cc737128280389964f3af9f511ff7b01",
    "brace_x_vision:shr4_disjoint:task_vector": "9a7db9b4e6aa26f3ad9f1f60a058311ef62dfc1c94189c2463041a45152d39cd",
    "brace_x_vision:shr6_clump_random:base_state": "d38bd0fb3db748fc8f3a434effade877b5e1502bc49569c082efd37cf2c7a921",
    "brace_x_vision:shr6_clump_random:ft_state": "1795215f278bb0d304daaf5f9f6bde426cb5820a008f779919dc0fad2ea2716e",
    "brace_x_vision:shr6_clump_random:layout": "50610d7a4a4eebed762a74b77e4c32a46d04f740ad958247a2e7bf37154f214e",
    "brace_x_vision:shr6_clump_random:task_vector": "254f3f268c25ca860e49faf3f957dec2c5564df6613c4762593761c411f128b1",
    "brace_x_vision:shr6_clump_top_bottom:base_state": "b344efece1c5f66a9c8d026e6e4e31cfc3f2b9d7d31446291487cc8b885ba7c5",
    "brace_x_vision:shr6_clump_top_bottom:ft_state": "3c9a0538b52d0d0b378fc13fffa988aacfefda96433fe39ed7a8af3d6444ed95",
    "brace_x_vision:shr6_clump_top_bottom:layout": "1c0695b0ca16441d4728a7d222041f14b2a97ed318799da384ea1ba05565c2b3",
    "brace_x_vision:shr6_clump_top_bottom:task_vector": "19028477b273ef3ff6474b3ac4f9f8ac679420eb18457e33149aeb3d67092b71",
    "brace_x_vision:shr6_density_clump:base_state": "b1ff87d9b577c24f79f9024470c05a133e1afc08331d74b0c573c66770f12ee8",
    "brace_x_vision:shr6_density_clump:ft_state": "6f4e7897c28f1fb4f094503be24e736515001b88bfe485b95963e4c0534a39f3",
    "brace_x_vision:shr6_density_clump:layout": "50610d7a4a4eebed762a74b77e4c32a46d04f740ad958247a2e7bf37154f214e",
    "brace_x_vision:shr6_density_clump:task_vector": "e65330b304ee522d7786576a9c0e1a0c9a7a05f8450447c6b527d9b5397e94bc",
    "brace_x_vision:shr6_density_spread_mod:base_state": "43b2ace70b2e7de7fcf4ec25e5c6edb64a294a88045e80408ddb5cede0baf4c7",
    "brace_x_vision:shr6_density_spread_mod:ft_state": "bf59451decb3a274e932dd58d269567b025a8be0b53be329661d83f2e37f0c8e",
    "brace_x_vision:shr6_density_spread_mod:layout": "4ba37d4b79d53c81e0f8aca7c0c82b6c006140549caa688bdabd6d8a64728f3e",
    "brace_x_vision:shr6_density_spread_mod:task_vector": "6bd4f7777a68c59daf9f60ee5add17c8daef79971d586934f5257b0a6a860d56",
    "brace_x_vision:shr6_disjoint:base_state": "a6f02e61dd9db1c6ed6f09dbb711bf85611cad913736e5139976373951f9b1f2",
    "brace_x_vision:shr6_disjoint:ft_state": "764fb850ebe92d56190ecd0f59819fb86342c18f71efbac85429ed8e721eb363",
    "brace_x_vision:shr6_disjoint:layout": "5cdc0c612282e04f8daeeff776887336125575d31ef8836981a63ee8a406a2ef",
    "brace_x_vision:shr6_disjoint:task_vector": "4859b58112821772b9cf4551a08638cb1c9c1b1492592bec6492cc36251b3b08",
    "brace_x_vision:shr6_disjoint_steer:base_state": "a6f02e61dd9db1c6ed6f09dbb711bf85611cad913736e5139976373951f9b1f2",
    "brace_x_vision:shr6_disjoint_steer:ft_state": "d5efa04fd04b80880d5c859a6a011916fb52a5ba62f8dfaae6b64e3b7ea84bd4",
    "brace_x_vision:shr6_disjoint_steer:layout": "5cdc0c612282e04f8daeeff776887336125575d31ef8836981a63ee8a406a2ef",
    "brace_x_vision:shr6_disjoint_steer:task_vector": "da9a8ef89ca795654264f58d33d61d68b2331b5f5ce5799865c492d3224d9ab1",
    "brace_x_vision:shr6_disjoint_top_bottom:base_state": "b123861fdf4bf17fe8d151ed5bea7cbcbb08210f09569e22ac20b5d9a804490f",
    "brace_x_vision:shr6_disjoint_top_bottom:ft_state": "dba8b7a9ee5f66be33d249e73c3153c9f4abe1ac9255ec430517ed832fedcf2d",
    "brace_x_vision:shr6_disjoint_top_bottom:layout": "15ffef8c682096f3069289cbed282aa0a834a3b3a964e943e9bf20bf07a47e60",
    "brace_x_vision:shr6_disjoint_top_bottom:task_vector": "f13104d2c32a95c03c31b2d61c4716b31d1c82c68d009ddc6ef125e8eb5171e7",
    "brace_x_vision:shr6_order_random:base_state": "b344efece1c5f66a9c8d026e6e4e31cfc3f2b9d7d31446291487cc8b885ba7c5",
    "brace_x_vision:shr6_order_random:ft_state": "3c9a0538b52d0d0b378fc13fffa988aacfefda96433fe39ed7a8af3d6444ed95",
    "brace_x_vision:shr6_order_random:layout": "1c0695b0ca16441d4728a7d222041f14b2a97ed318799da384ea1ba05565c2b3",
    "brace_x_vision:shr6_order_random:task_vector": "19028477b273ef3ff6474b3ac4f9f8ac679420eb18457e33149aeb3d67092b71",
    "brace_x_vision:shr6_order_top_bottom:base_state": "d5e108f26af9b5d06827cc9426f7c2454341ecbd1bcf239a680fcc515c35be5f",
    "brace_x_vision:shr6_order_top_bottom:ft_state": "1109d9e543c5039da1c0e4c940f79b3f2b6cf0dbdd854f1dfe21b034bab04f56",
    "brace_x_vision:shr6_order_top_bottom:layout": "bc60a3c802358f82e7bc4ed72d3ed875ec87fdc7b44907d37ee20f8dbd22853e",
    "brace_x_vision:shr6_order_top_bottom:task_vector": "2f524f54767a777f15097d514f3edf440717783fe2144f80ffbe92b36e01e2ac",
    "brace_x_vision:shr6_spread:base_state": "b2076365cced763d61958815bb1c0166eba78a3b163f74d7f1877393c17ed42b",
    "brace_x_vision:shr6_spread:ft_state": "be8d23e4d58d883cc9691d2bed5a716ad9d69843e174ea9a08d81768bd7d42ef",
    "brace_x_vision:shr6_spread:layout": "1b1fb0ab68216cf9c323efc849bc144c0f940ea75fd4f85e84771758501597ab",
    "brace_x_vision:shr6_spread:task_vector": "7dac308a800bc0fddb4c1fa6bad960ec722e6985651d145f24a285ee933c24aa",
    "brace_x_vision:shr_cascade_iters2:base_state": "b177d0854c5eedd8359d60d507c23d9a6498ace9fca85dca9dd7a52681ef1956",
    "brace_x_vision:shr_cascade_iters2:ft_state": "998b29cd02c39d244b5160048d88b45ff2ad9dcebab8765f39b81884126d107b",
    "brace_x_vision:shr_cascade_iters2:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_cascade_iters2:task_vector": "45e7205042fade0c57a287d963d71df3ffe04226806460af6b57007eeb8130bb",
    "brace_x_vision:shr_component_ridge:base_state": "e1d2d289d3754fef552c68445dee53aa49b8b3f5529a8ea754edeead26a605f2",
    "brace_x_vision:shr_component_ridge:ft_state": "cd17d966ed428dcdee3a98080f3a9576ee6677a5c30e5318745812f3a4937f61",
    "brace_x_vision:shr_component_ridge:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_component_ridge:task_vector": "2757ec5d77a2ff0c4c4b13d54f5b6eac94da1b29abe7eced4ee0c09d4a7beaa2",
    "brace_x_vision:shr_dampening:base_state": "9ca271a5e0e13b37ea56aabfc273c037415b58e5fd176ad598fc1d290805630c",
    "brace_x_vision:shr_dampening:ft_state": "cd1696f5c4d99ab548a3014a0973f2c74309ec08ba9255cb5b59408179acc10c",
    "brace_x_vision:shr_dampening:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_dampening:task_vector": "3060e83ffe35a78c0d1f10eb16d0f3a2471507fe5fae3aed7eb1f7c2bfdfac4e",
    "brace_x_vision:shr_diag_independent:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_diag_independent:diag": "3de0f807a1bc488911fef8c8f28ae1a8e18c4f45300991bb75c70ec91b572e4b",
    "brace_x_vision:shr_diag_independent:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_x_vision:shr_diag_independent:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_diag_independent:task_vector": "1afc337731bd4e81e4ca6c11a4499fa916a4fdfebd26ba2f8e8d79cd8303fa93",
    "brace_x_vision:shr_diag_shared:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_diag_shared:diag": "065233af61d761101d728bb6dcc12d46ce74f0d088f571c91f35430745d5b59a",
    "brace_x_vision:shr_diag_shared:ft_state": "19ec6af8c7cd4de95a2d2a9b683c43bbc09290e162780eaed618870ce4c051d5",
    "brace_x_vision:shr_diag_shared:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_diag_shared:task_vector": "3b8f2d5e6301f26e92116a4ec174a702122cc710d68042093f973d570af8894f",
    "brace_x_vision:shr_diag_shared_ft:base_state": "56169469a92385efd720e719880d8da79b2220f92cd2b34981b5bc58db0f2e4d",
    "brace_x_vision:shr_diag_shared_ft:diag": "4af1692abeb9687c8e50ee74d7fca482e87db5237efb69c50a65144d873e88d5",
    "brace_x_vision:shr_diag_shared_ft:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_x_vision:shr_diag_shared_ft:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_diag_shared_ft:task_vector": "60ced8b4cb1274fb82a04fff3ca090e38a3ac23444badb371da365f60590968c",
    "brace_x_vision:shr_diag_steer:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_diag_steer:diag": "e69abd851e643dbf0eaae1bcdc908414754a23738baadea7604ef00e557697d8",
    "brace_x_vision:shr_diag_steer:ft_state": "6f28aed34c136fd849389d248e94ad77cafded2976d9d0abc8315033a0360e7a",
    "brace_x_vision:shr_diag_steer:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_diag_steer:task_vector": "7bf901a6c7ecba2e0cacdd92c285c037c1f0085ea39be1b0fd0927ccf086441d",
    "brace_x_vision:shr_dup:base_state": "065d944b9986bd8e7c1f76402ee6ce1dca193519754c855ea89bf8e204c9206a",
    "brace_x_vision:shr_dup:ft_state": "8fe69e1bc7c8b235f5888cd1ea566c008816bdf3f2870497b7d3a9afd3f9fe9e",
    "brace_x_vision:shr_dup:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_dup:task_vector": "fff536173409c8730e74cc2dea750178c26dd9c752519b7b9e42987e7ed7247e",
    "brace_x_vision:shr_eager:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_eager:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_x_vision:shr_eager:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_eager:task_vector": "1afc337731bd4e81e4ca6c11a4499fa916a4fdfebd26ba2f8e8d79cd8303fa93",
    "brace_x_vision:shr_ridge_weight:base_state": "46772e0e5501f2e3e821d82e172a37b3197b4f532ae811b2e37fa470667a83ee",
    "brace_x_vision:shr_ridge_weight:ft_state": "0e512368b5a61531d20dded95e33d8cb3ccede83cbec2fe2e187c2d8801c7946",
    "brace_x_vision:shr_ridge_weight:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_ridge_weight:task_vector": "a403c4fdf21e46b09494fe89d16e9f84af38cf23461581f341b93033bd2feb54",
    "brace_x_vision:shr_share_ft_refs:base_state": "7cdf3a5a744d253cf3a4ae95bdce2bd365b1f23e76558a7b6b3f3b0045fa4f90",
    "brace_x_vision:shr_share_ft_refs:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_x_vision:shr_share_ft_refs:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_share_ft_refs:task_vector": "8ed15a24281a2479e31b86db125ab9a2a174db7e72415c8f73e69b6f3e8c9b3a",
    "brace_x_vision:shr_skip_correction:base_state": "9e1fd6cb1ea00b3969a8e8373ceba9002360b7cf76378c3fc8c4bf4175978fd4",
    "brace_x_vision:shr_skip_correction:ft_state": "12dc2cc5844580dd90b07277d8009c3f4ffc19bf6e61a986f3102daff53c1fcd",
    "brace_x_vision:shr_skip_correction:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_skip_correction:task_vector": "5851291766515fcd90c8aea3ba334c89de8842c8c5fb09ca6fff59def4eaa476",
    "brace_x_vision:shr_skip_final_ln:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_skip_final_ln:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_x_vision:shr_skip_final_ln:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_skip_final_ln:task_vector": "1afc337731bd4e81e4ca6c11a4499fa916a4fdfebd26ba2f8e8d79cd8303fa93",
    "brace_x_vision:shr_steer:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_x_vision:shr_steer:ft_state": "6f28aed34c136fd849389d248e94ad77cafded2976d9d0abc8315033a0360e7a",
    "brace_x_vision:shr_steer:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_x_vision:shr_steer:task_vector": "7bf901a6c7ecba2e0cacdd92c285c037c1f0085ea39be1b0fd0927ccf086441d",
}

TABLES: dict = {
    "balanced_collapse_spans": {
        "0,1,bottom-top": "ERR ValueError",
        "0,1,random": "ERR ValueError",
        "0,1,top-bottom": "ERR ValueError",
        "1,1,bottom-top": [(0,)],
        "1,1,random": "ERR ValueError",
        "1,1,top-bottom": [(0,)],
        "2,1,bottom-top": [(0, 1)],
        "2,1,random": "ERR ValueError",
        "2,1,top-bottom": [(0, 1)],
        "2,2,bottom-top": [(0,), (1,)],
        "2,2,random": "ERR ValueError",
        "2,2,top-bottom": [(0,), (1,)],
        "3,2,bottom-top": [(0,), (1, 2)],
        "3,2,random": "ERR ValueError",
        "3,2,top-bottom": [(0, 1), (2,)],
        "3,4,bottom-top": "ERR ValueError",
        "3,4,random": "ERR ValueError",
        "3,4,top-bottom": "ERR ValueError",
        "4,2,bottom-top": [(0, 1), (2, 3)],
        "4,2,random": "ERR ValueError",
        "4,2,top-bottom": [(0, 1), (2, 3)],
        "5,3,bottom-top": [(0,), (1, 2), (3, 4)],
        "5,3,random": "ERR ValueError",
        "5,3,top-bottom": [(0, 1), (2, 3), (4,)],
        "6,4,bottom-top": [(0,), (1, 2), (3,), (4, 5)],
        "6,4,random": "ERR ValueError",
        "6,4,top-bottom": [(0, 1), (2,), (3, 4), (5,)],
        "7,3,bottom-top": [(0, 1), (2, 3), (4, 5, 6)],
        "7,3,random": "ERR ValueError",
        "7,3,top-bottom": [(0, 1, 2), (3, 4), (5, 6)],
        "8,3,bottom-top": [(0, 1), (2, 3, 4), (5, 6, 7)],
        "8,3,random": "ERR ValueError",
        "8,3,top-bottom": [(0, 1, 2), (3, 4, 5), (6, 7)],
    },
    "build_extension_layout": {
        "2:0+1+1": {
            "direction": "extend",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0},
                {"block_kind": "inserted", "position": 1, "source_orig_idx": 0},
                {"block_kind": "original", "position": 2, "source_orig_idx": 1},
                {"block_kind": "inserted", "position": 3, "source_orig_idx": 1},
                {"block_kind": "inserted", "position": 4, "source_orig_idx": 1},
            ),
            "final_depth": 5,
            "inserted_blocks": (
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 2,
                    "position": 1,
                    "source_orig_idx": 0,
                    "source_position": 0,
                },
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 2,
                    "position": 3,
                    "source_orig_idx": 1,
                    "source_position": 2,
                },
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 2,
                    "position": 4,
                    "source_orig_idx": 1,
                    "source_position": 2,
                },
            ),
            "original_positions": {0: 0, 1: 2},
            "p1_source_ancestry": "inserted_position",
        },
        "3:0+1": {
            "direction": "extend",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0},
                {"block_kind": "inserted", "position": 1, "source_orig_idx": 0},
                {"block_kind": "original", "position": 2, "source_orig_idx": 1},
                {"block_kind": "inserted", "position": 3, "source_orig_idx": 1},
                {"block_kind": "original", "position": 4, "source_orig_idx": 2},
            ),
            "final_depth": 5,
            "inserted_blocks": (
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 2,
                    "position": 1,
                    "source_orig_idx": 0,
                    "source_position": 0,
                },
                {
                    "neighbour_orig_idx": 2,
                    "neighbour_position": 4,
                    "position": 3,
                    "source_orig_idx": 1,
                    "source_position": 2,
                },
            ),
            "original_positions": {0: 0, 1: 2, 2: 4},
            "p1_source_ancestry": "inserted_position",
        },
        "3:2": {
            "direction": "extend",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0},
                {"block_kind": "original", "position": 1, "source_orig_idx": 1},
                {"block_kind": "original", "position": 2, "source_orig_idx": 2},
                {"block_kind": "inserted", "position": 3, "source_orig_idx": 2},
            ),
            "final_depth": 4,
            "inserted_blocks": (
                {
                    "neighbour_orig_idx": 2,
                    "neighbour_position": 2,
                    "position": 3,
                    "source_orig_idx": 2,
                    "source_position": 2,
                },
            ),
            "original_positions": {0: 0, 1: 1, 2: 2},
            "p1_source_ancestry": "inserted_position",
        },
        "4:0+0+3": {
            "direction": "extend",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0},
                {"block_kind": "inserted", "position": 1, "source_orig_idx": 0},
                {"block_kind": "inserted", "position": 2, "source_orig_idx": 0},
                {"block_kind": "original", "position": 3, "source_orig_idx": 1},
                {"block_kind": "original", "position": 4, "source_orig_idx": 2},
                {"block_kind": "original", "position": 5, "source_orig_idx": 3},
                {"block_kind": "inserted", "position": 6, "source_orig_idx": 3},
            ),
            "final_depth": 7,
            "inserted_blocks": (
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 3,
                    "position": 1,
                    "source_orig_idx": 0,
                    "source_position": 0,
                },
                {
                    "neighbour_orig_idx": 1,
                    "neighbour_position": 3,
                    "position": 2,
                    "source_orig_idx": 0,
                    "source_position": 0,
                },
                {
                    "neighbour_orig_idx": 3,
                    "neighbour_position": 5,
                    "position": 6,
                    "source_orig_idx": 3,
                    "source_position": 5,
                },
            ),
            "original_positions": {0: 0, 1: 3, 2: 4, 3: 5},
            "p1_source_ancestry": "inserted_position",
        },
    },
    "build_reduction_layout": {
        "all_but_last": {
            "direction": "shrink",
            "final_blocks": (
                {"block_kind": "collapsed", "position": 0, "source_orig_idx": 2, "span_orig_idxs": (0, 1, 2)},
                {"block_kind": "original", "position": 1, "source_orig_idx": 3, "span_orig_idxs": (3,)},
            ),
            "final_depth": 2,
            "inserted_blocks": (),
            "original_positions": {0: 0, 1: 0, 2: 0, 3: 1},
            "p1_source_ancestry": "span_end_boundary",
        },
        "head_merged": {
            "direction": "shrink",
            "final_blocks": (
                {"block_kind": "collapsed", "position": 0, "source_orig_idx": 1, "span_orig_idxs": (0, 1)},
                {"block_kind": "original", "position": 1, "source_orig_idx": 2, "span_orig_idxs": (2,)},
                {"block_kind": "original", "position": 2, "source_orig_idx": 3, "span_orig_idxs": (3,)},
            ),
            "final_depth": 3,
            "inserted_blocks": (),
            "original_positions": {0: 0, 1: 0, 2: 1, 3: 2},
            "p1_source_ancestry": "span_end_boundary",
        },
        "mid_merged": {
            "direction": "shrink",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0, "span_orig_idxs": (0,)},
                {"block_kind": "collapsed", "position": 1, "source_orig_idx": 2, "span_orig_idxs": (1, 2)},
                {"block_kind": "original", "position": 2, "source_orig_idx": 3, "span_orig_idxs": (3,)},
            ),
            "final_depth": 3,
            "inserted_blocks": (),
            "original_positions": {0: 0, 1: 1, 2: 1, 3: 2},
            "p1_source_ancestry": "span_end_boundary",
        },
        "singletons4": {
            "direction": "shrink",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0, "span_orig_idxs": (0,)},
                {"block_kind": "original", "position": 1, "source_orig_idx": 1, "span_orig_idxs": (1,)},
                {"block_kind": "original", "position": 2, "source_orig_idx": 2, "span_orig_idxs": (2,)},
                {"block_kind": "original", "position": 3, "source_orig_idx": 3, "span_orig_idxs": (3,)},
            ),
            "final_depth": 4,
            "inserted_blocks": (),
            "original_positions": {0: 0, 1: 1, 2: 2, 3: 3},
            "p1_source_ancestry": "span_end_boundary",
        },
        "tail_merged": {
            "direction": "shrink",
            "final_blocks": (
                {"block_kind": "original", "position": 0, "source_orig_idx": 0, "span_orig_idxs": (0,)},
                {"block_kind": "original", "position": 1, "source_orig_idx": 1, "span_orig_idxs": (1,)},
                {"block_kind": "collapsed", "position": 2, "source_orig_idx": 3, "span_orig_idxs": (2, 3)},
            ),
            "final_depth": 3,
            "inserted_blocks": (),
            "original_positions": {0: 0, 1: 1, 2: 2, 3: 2},
            "p1_source_ancestry": "span_end_boundary",
        },
    },
    "collapse_schedule:decoder": {
        "1,1,bottom-top,clump": "ERR ValueError",
        "1,1,bottom-top,spread": "ERR ValueError",
        "1,1,bottom-top,spread_mod": "ERR ValueError",
        "1,1,random,clump": "ERR ValueError",
        "1,1,random,spread": "ERR ValueError",
        "1,1,random,spread_mod": "ERR ValueError",
        "1,1,top-bottom,clump": "ERR ValueError",
        "1,1,top-bottom,spread": "ERR ValueError",
        "1,1,top-bottom,spread_mod": "ERR ValueError",
        "2,1,bottom-top,clump": [0],
        "2,1,bottom-top,spread": [0],
        "2,1,bottom-top,spread_mod": [0],
        "2,1,random,clump": [0],
        "2,1,random,spread": [0],
        "2,1,random,spread_mod": [0],
        "2,1,top-bottom,clump": [0],
        "2,1,top-bottom,spread": [0],
        "2,1,top-bottom,spread_mod": [0],
        "3,1,bottom-top,clump": [0],
        "3,1,bottom-top,spread": [0],
        "3,1,bottom-top,spread_mod": [0],
        "3,1,random,clump": [1],
        "3,1,random,spread": [0],
        "3,1,random,spread_mod": [0],
        "3,1,top-bottom,clump": [1],
        "3,1,top-bottom,spread": [1],
        "3,1,top-bottom,spread_mod": [0],
        "3,2,bottom-top,clump": [0, 0],
        "3,2,bottom-top,spread": [0, 1],
        "3,2,bottom-top,spread_mod": [0, 1],
        "3,2,random,clump": [1, 0],
        "3,2,random,spread": [0, 1],
        "3,2,random,spread_mod": [0, 1],
        "3,2,top-bottom,clump": [1, 1],
        "3,2,top-bottom,spread": [1, 0],
        "3,2,top-bottom,spread_mod": [0, 1],
        "4,1,bottom-top,clump": [0],
        "4,1,bottom-top,spread": [0],
        "4,1,bottom-top,spread_mod": [0],
        "4,1,random,clump": [2],
        "4,1,random,spread": [0],
        "4,1,random,spread_mod": [0],
        "4,1,top-bottom,clump": [2],
        "4,1,top-bottom,spread": [2],
        "4,1,top-bottom,spread_mod": [0],
        "4,2,bottom-top,clump": [0, 0],
        "4,2,bottom-top,spread": [0, 1],
        "4,2,bottom-top,spread_mod": [0, 1],
        "4,2,random,clump": [2, 1],
        "4,2,random,spread": [0, 1],
        "4,2,random,spread_mod": [0, 1],
        "4,2,top-bottom,clump": [2, 2],
        "4,2,top-bottom,spread": [2, 1],
        "4,2,top-bottom,spread_mod": [0, 1],
        "4,3,bottom-top,clump": [0, 0, 0],
        "4,3,bottom-top,spread": [0, 1, 2],
        "4,3,bottom-top,spread_mod": [0, 1, 2],
        "4,3,random,clump": [2, 1, 0],
        "4,3,random,spread": [0, 1, 2],
        "4,3,random,spread_mod": [0, 1, 2],
        "4,3,top-bottom,clump": [2, 2, 2],
        "4,3,top-bottom,spread": [2, 1, 0],
        "4,3,top-bottom,spread_mod": [0, 1, 2],
        "6,2,bottom-top,clump": [0, 0],
        "6,2,bottom-top,spread": [0, 2],
        "6,2,bottom-top,spread_mod": [0, 1],
        "6,2,random,clump": [1, 0],
        "6,2,random,spread": [4, 3],
        "6,2,random,spread_mod": [0, 1],
        "6,2,top-bottom,clump": [4, 4],
        "6,2,top-bottom,spread": [4, 2],
        "6,2,top-bottom,spread_mod": [0, 1],
        "6,3,bottom-top,clump": [0, 0, 0],
        "6,3,bottom-top,spread": [0, 1, 3],
        "6,3,bottom-top,spread_mod": [0, 1, 2],
        "6,3,random,clump": [1, 0, 2],
        "6,3,random,spread": [4, 3, 2],
        "6,3,random,spread_mod": [0, 1, 2],
        "6,3,top-bottom,clump": [4, 4, 4],
        "6,3,top-bottom,spread": [4, 3, 1],
        "6,3,top-bottom,spread_mod": [0, 1, 2],
        "6,5,bottom-top,clump": [0, 0, 0, 0, 0],
        "6,5,bottom-top,spread": [0, 1, 2, 3, 4],
        "6,5,bottom-top,spread_mod": [0, 1, 2, 3, 4],
        "6,5,random,clump": [1, 0, 2, 2, 1],
        "6,5,random,spread": [4, 3, 2, 0, 1],
        "6,5,random,spread_mod": [0, 1, 2, 3, 4],
        "6,5,top-bottom,clump": [4, 4, 4, 4, 4],
        "6,5,top-bottom,spread": [4, 3, 2, 1, 0],
        "6,5,top-bottom,spread_mod": [0, 1, 2, 3, 4],
        "8,3,bottom-top,clump": [0, 0, 0],
        "8,3,bottom-top,spread": [0, 2, 4],
        "8,3,bottom-top,spread_mod": [0, 1, 2],
        "8,3,random,clump": [6, 1, 0],
        "8,3,random,spread": [4, 5, 3],
        "8,3,random,spread_mod": [0, 1, 2],
        "8,3,top-bottom,clump": [6, 6, 6],
        "8,3,top-bottom,spread": [6, 4, 2],
        "8,3,top-bottom,spread_mod": [0, 1, 2],
        "8,4,bottom-top,clump": [0, 0, 0, 0],
        "8,4,bottom-top,spread": [0, 1, 3, 5],
        "8,4,bottom-top,spread_mod": [0, 1, 2, 3],
        "8,4,random,clump": [6, 1, 0, 2],
        "8,4,random,spread": [4, 5, 3, 2],
        "8,4,random,spread_mod": [0, 1, 2, 3],
        "8,4,top-bottom,clump": [6, 6, 6, 6],
        "8,4,top-bottom,spread": [6, 5, 3, 1],
        "8,4,top-bottom,spread_mod": [0, 1, 2, 3],
    },
    "collapse_schedule:vision": {
        "1,1,bottom-top,clump": "ERR ValueError",
        "1,1,bottom-top,spread": "ERR ValueError",
        "1,1,bottom-top,spread_mod": "ERR ValueError",
        "1,1,random,clump": "ERR ValueError",
        "1,1,random,spread": "ERR ValueError",
        "1,1,random,spread_mod": "ERR ValueError",
        "1,1,top-bottom,clump": "ERR ValueError",
        "1,1,top-bottom,spread": "ERR ValueError",
        "1,1,top-bottom,spread_mod": "ERR ValueError",
        "2,1,bottom-top,clump": [0],
        "2,1,bottom-top,spread": [0],
        "2,1,bottom-top,spread_mod": [0],
        "2,1,random,clump": [0],
        "2,1,random,spread": [0],
        "2,1,random,spread_mod": [0],
        "2,1,top-bottom,clump": [0],
        "2,1,top-bottom,spread": [0],
        "2,1,top-bottom,spread_mod": [0],
        "3,1,bottom-top,clump": [0],
        "3,1,bottom-top,spread": [0],
        "3,1,bottom-top,spread_mod": [0],
        "3,1,random,clump": [1],
        "3,1,random,spread": [0],
        "3,1,random,spread_mod": [0],
        "3,1,top-bottom,clump": [1],
        "3,1,top-bottom,spread": [1],
        "3,1,top-bottom,spread_mod": [1],
        "3,2,bottom-top,clump": [0, 0],
        "3,2,bottom-top,spread": [0, 1],
        "3,2,bottom-top,spread_mod": [0, 1],
        "3,2,random,clump": [1, 0],
        "3,2,random,spread": [0, 1],
        "3,2,random,spread_mod": [0, 1],
        "3,2,top-bottom,clump": [1, 1],
        "3,2,top-bottom,spread": [1, 0],
        "3,2,top-bottom,spread_mod": [1, 0],
        "4,1,bottom-top,clump": [0],
        "4,1,bottom-top,spread": [0],
        "4,1,bottom-top,spread_mod": [0],
        "4,1,random,clump": [2],
        "4,1,random,spread": [0],
        "4,1,random,spread_mod": [0],
        "4,1,top-bottom,clump": [2],
        "4,1,top-bottom,spread": [2],
        "4,1,top-bottom,spread_mod": [2],
        "4,2,bottom-top,clump": [0, 0],
        "4,2,bottom-top,spread": [0, 1],
        "4,2,bottom-top,spread_mod": [0, 2],
        "4,2,random,clump": [2, 1],
        "4,2,random,spread": [0, 1],
        "4,2,random,spread_mod": [0, 2],
        "4,2,top-bottom,clump": [2, 2],
        "4,2,top-bottom,spread": [2, 1],
        "4,2,top-bottom,spread_mod": [2, 0],
        "4,3,bottom-top,clump": [0, 0, 0],
        "4,3,bottom-top,spread": [0, 1, 2],
        "4,3,bottom-top,spread_mod": [0, 1, 2],
        "4,3,random,clump": [2, 1, 0],
        "4,3,random,spread": [0, 1, 2],
        "4,3,random,spread_mod": [0, 1, 2],
        "4,3,top-bottom,clump": [2, 2, 2],
        "4,3,top-bottom,spread": [2, 1, 0],
        "4,3,top-bottom,spread_mod": [2, 1, 0],
        "6,2,bottom-top,clump": [0, 0],
        "6,2,bottom-top,spread": [0, 2],
        "6,2,bottom-top,spread_mod": [0, 4],
        "6,2,random,clump": [1, 0],
        "6,2,random,spread": [4, 3],
        "6,2,random,spread_mod": [0, 4],
        "6,2,top-bottom,clump": [4, 4],
        "6,2,top-bottom,spread": [4, 2],
        "6,2,top-bottom,spread_mod": [4, 0],
        "6,3,bottom-top,clump": [0, 0, 0],
        "6,3,bottom-top,spread": [0, 1, 3],
        "6,3,bottom-top,spread_mod": [0, 2, 4],
        "6,3,random,clump": [1, 0, 2],
        "6,3,random,spread": [4, 3, 2],
        "6,3,random,spread_mod": [0, 2, 4],
        "6,3,top-bottom,clump": [4, 4, 4],
        "6,3,top-bottom,spread": [4, 3, 1],
        "6,3,top-bottom,spread_mod": [4, 2, 0],
        "6,5,bottom-top,clump": [0, 0, 0, 0, 0],
        "6,5,bottom-top,spread": [0, 1, 2, 3, 4],
        "6,5,bottom-top,spread_mod": [0, 1, 2, 3, 4],
        "6,5,random,clump": [1, 0, 2, 2, 1],
        "6,5,random,spread": [4, 3, 2, 0, 1],
        "6,5,random,spread_mod": [4, 3, 2, 0, 1],
        "6,5,top-bottom,clump": [4, 4, 4, 4, 4],
        "6,5,top-bottom,spread": [4, 3, 2, 1, 0],
        "6,5,top-bottom,spread_mod": [4, 3, 2, 1, 0],
        "8,3,bottom-top,clump": [0, 0, 0],
        "8,3,bottom-top,spread": [0, 2, 4],
        "8,3,bottom-top,spread_mod": [0, 3, 6],
        "8,3,random,clump": [6, 1, 0],
        "8,3,random,spread": [4, 5, 3],
        "8,3,random,spread_mod": [0, 3, 6],
        "8,3,top-bottom,clump": [6, 6, 6],
        "8,3,top-bottom,spread": [6, 4, 2],
        "8,3,top-bottom,spread_mod": [6, 3, 0],
        "8,4,bottom-top,clump": [0, 0, 0, 0],
        "8,4,bottom-top,spread": [0, 1, 3, 5],
        "8,4,bottom-top,spread_mod": [0, 2, 4, 6],
        "8,4,random,clump": [6, 1, 0, 2],
        "8,4,random,spread": [4, 5, 3, 2],
        "8,4,random,spread_mod": [0, 2, 4, 6],
        "8,4,top-bottom,clump": [6, 6, 6, 6],
        "8,4,top-bottom,spread": [6, 5, 3, 1],
        "8,4,top-bottom,spread_mod": [6, 4, 2, 0],
    },
    "disjoint_collapse_schedule": {
        "1,1,bottom-top": "ERR ValueError",
        "1,1,random": "ERR ValueError",
        "1,1,top-bottom": "ERR ValueError",
        "2,1,bottom-top": [0],
        "2,1,random": "ERR ValueError",
        "2,1,top-bottom": [0],
        "3,1,bottom-top": [1],
        "3,1,random": "ERR ValueError",
        "3,1,top-bottom": [0],
        "3,2,bottom-top": [0, 0],
        "3,2,random": "ERR ValueError",
        "3,2,top-bottom": [0, 0],
        "4,1,bottom-top": [2],
        "4,1,random": "ERR ValueError",
        "4,1,top-bottom": [0],
        "4,2,bottom-top": [0, 2],
        "4,2,random": "ERR ValueError",
        "4,2,top-bottom": [0, 2],
        "4,3,bottom-top": [0, 0, 0],
        "4,3,random": "ERR ValueError",
        "4,3,top-bottom": [0, 0, 0],
        "6,2,bottom-top": [1, 4],
        "6,2,random": "ERR ValueError",
        "6,2,top-bottom": [0, 3],
        "6,3,bottom-top": [0, 2, 4],
        "6,3,random": "ERR ValueError",
        "6,3,top-bottom": [0, 2, 4],
        "6,5,bottom-top": [0, 0, 0, 0, 0],
        "6,5,random": "ERR ValueError",
        "6,5,top-bottom": [0, 0, 0, 0, 0],
        "8,3,bottom-top": [1, 4, 6],
        "8,3,random": "ERR ValueError",
        "8,3,top-bottom": [0, 2, 5],
        "8,4,bottom-top": [0, 2, 4, 6],
        "8,4,random": "ERR ValueError",
        "8,4,top-bottom": [0, 2, 4, 6],
    },
    "dup_schedule:decoder": {
        "1,1,bottom-top,clump": [0],
        "1,1,bottom-top,spread": [0],
        "1,1,bottom-top,spread_mod": "ERR ZeroDivisionError",
        "1,1,random,clump": [0],
        "1,1,random,spread": [0],
        "1,1,random,spread_mod": "ERR ZeroDivisionError",
        "1,1,top-bottom,clump": [0],
        "1,1,top-bottom,spread": [0],
        "1,1,top-bottom,spread_mod": "ERR ZeroDivisionError",
        "2,1,bottom-top,clump": [0],
        "2,1,bottom-top,spread": [0],
        "2,1,bottom-top,spread_mod": [0],
        "2,1,random,clump": [0],
        "2,1,random,spread": [0],
        "2,1,random,spread_mod": [0],
        "2,1,top-bottom,clump": [1],
        "2,1,top-bottom,spread": [0],
        "2,1,top-bottom,spread_mod": [0],
        "2,2,bottom-top,clump": [0, 0],
        "2,2,bottom-top,spread": [0, 1],
        "2,2,bottom-top,spread_mod": [0, 0],
        "2,2,random,clump": [0, 0],
        "2,2,random,spread": [1, 0],
        "2,2,random,spread_mod": [0, 0],
        "2,2,top-bottom,clump": [1, 1],
        "2,2,top-bottom,spread": [1, 0],
        "2,2,top-bottom,spread_mod": [0, 0],
        "2,3,bottom-top,clump": [0, 0, 0],
        "2,3,bottom-top,spread": [0, 0, 1],
        "2,3,bottom-top,spread_mod": [0, 0, 0],
        "2,3,random,clump": [0, 0, 0],
        "2,3,random,spread": [1, 0, 0],
        "2,3,random,spread_mod": [0, 0, 0],
        "2,3,top-bottom,clump": [1, 1, 1],
        "2,3,top-bottom,spread": [1, 1, 0],
        "2,3,top-bottom,spread_mod": [0, 0, 0],
        "3,1,bottom-top,clump": [0],
        "3,1,bottom-top,spread": [0],
        "3,1,bottom-top,spread_mod": [0],
        "3,1,random,clump": [0],
        "3,1,random,spread": [1],
        "3,1,random,spread_mod": [0],
        "3,1,top-bottom,clump": [2],
        "3,1,top-bottom,spread": [1],
        "3,1,top-bottom,spread_mod": [0],
        "3,2,bottom-top,clump": [0, 0],
        "3,2,bottom-top,spread": [0, 1],
        "3,2,bottom-top,spread_mod": [0, 1],
        "3,2,random,clump": [0, 0],
        "3,2,random,spread": [1, 0],
        "3,2,random,spread_mod": [0, 1],
        "3,2,top-bottom,clump": [2, 2],
        "3,2,top-bottom,spread": [1, 0],
        "3,2,top-bottom,spread_mod": [0, 1],
        "3,5,bottom-top,clump": [0, 0, 0, 0, 0],
        "3,5,bottom-top,spread": [0, 0, 1, 1, 2],
        "3,5,bottom-top,spread_mod": [0, 1, 0, 1, 0],
        "3,5,random,clump": [0, 0, 0, 0, 0],
        "3,5,random,spread": [1, 2, 0, 0, 1],
        "3,5,random,spread_mod": [0, 1, 0, 1, 0],
        "3,5,top-bottom,clump": [2, 2, 2, 2, 2],
        "3,5,top-bottom,spread": [2, 2, 1, 1, 0],
        "3,5,top-bottom,spread_mod": [0, 1, 0, 1, 0],
        "4,2,bottom-top,clump": [0, 0],
        "4,2,bottom-top,spread": [0, 1],
        "4,2,bottom-top,spread_mod": [0, 1],
        "4,2,random,clump": [0, 0],
        "4,2,random,spread": [1, 2],
        "4,2,random,spread_mod": [0, 1],
        "4,2,top-bottom,clump": [3, 3],
        "4,2,top-bottom,spread": [2, 1],
        "4,2,top-bottom,spread_mod": [0, 1],
        "4,4,bottom-top,clump": [0, 0, 0, 0],
        "4,4,bottom-top,spread": [0, 1, 2, 3],
        "4,4,bottom-top,spread_mod": [0, 1, 2, 0],
        "4,4,random,clump": [0, 0, 0, 0],
        "4,4,random,spread": [1, 3, 2, 0],
        "4,4,random,spread_mod": [0, 1, 2, 0],
        "4,4,top-bottom,clump": [3, 3, 3, 3],
        "4,4,top-bottom,spread": [3, 2, 1, 0],
        "4,4,top-bottom,spread_mod": [0, 1, 2, 0],
        "4,6,bottom-top,clump": [0, 0, 0, 0, 0, 0],
        "4,6,bottom-top,spread": [0, 0, 1, 2, 2, 3],
        "4,6,bottom-top,spread_mod": [0, 1, 2, 0, 1, 2],
        "4,6,random,clump": [0, 0, 0, 0, 0, 0],
        "4,6,random,spread": [1, 3, 2, 0, 0, 2],
        "4,6,random,spread_mod": [0, 1, 2, 0, 1, 2],
        "4,6,top-bottom,clump": [3, 3, 3, 3, 3, 3],
        "4,6,top-bottom,spread": [3, 3, 2, 1, 1, 0],
        "4,6,top-bottom,spread_mod": [0, 1, 2, 0, 1, 2],
        "6,3,bottom-top,clump": [0, 0, 0],
        "6,3,bottom-top,spread": [0, 1, 3],
        "6,3,bottom-top,spread_mod": [0, 1, 2],
        "6,3,random,clump": [4, 4, 4],
        "6,3,random,spread": [0, 2, 4],
        "6,3,random,spread_mod": [0, 1, 2],
        "6,3,top-bottom,clump": [5, 5, 5],
        "6,3,top-bottom,spread": [4, 3, 1],
        "6,3,top-bottom,spread_mod": [0, 1, 2],
        "6,8,bottom-top,clump": [0, 0, 0, 0, 0, 0, 0, 0],
        "6,8,bottom-top,spread": [0, 0, 1, 2, 3, 3, 4, 5],
        "6,8,bottom-top,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "6,8,random,clump": [4, 4, 4, 4, 4, 4, 4, 4],
        "6,8,random,spread": [4, 0, 2, 5, 3, 1, 1, 3],
        "6,8,random,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "6,8,top-bottom,clump": [5, 5, 5, 5, 5, 5, 5, 5],
        "6,8,top-bottom,spread": [5, 5, 4, 3, 2, 2, 1, 0],
        "6,8,top-bottom,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "8,3,bottom-top,clump": [0, 0, 0],
        "8,3,bottom-top,spread": [0, 2, 4],
        "8,3,bottom-top,spread_mod": [0, 1, 2],
        "8,3,random,clump": [4, 4, 4],
        "8,3,random,spread": [4, 0, 2],
        "8,3,random,spread_mod": [0, 1, 2],
        "8,3,top-bottom,clump": [7, 7, 7],
        "8,3,top-bottom,spread": [6, 4, 2],
        "8,3,top-bottom,spread_mod": [0, 1, 2],
    },
    "dup_schedule:vision": {
        "1,1,bottom-top,clump": [0],
        "1,1,bottom-top,spread": [0],
        "1,1,bottom-top,spread_mod": "ERR ZeroDivisionError",
        "1,1,random,clump": [0],
        "1,1,random,spread": [0],
        "1,1,random,spread_mod": "ERR ZeroDivisionError",
        "1,1,top-bottom,clump": [0],
        "1,1,top-bottom,spread": [0],
        "1,1,top-bottom,spread_mod": "ERR ZeroDivisionError",
        "2,1,bottom-top,clump": [0],
        "2,1,bottom-top,spread": [0],
        "2,1,bottom-top,spread_mod": [0],
        "2,1,random,clump": [0],
        "2,1,random,spread": [0],
        "2,1,random,spread_mod": [0],
        "2,1,top-bottom,clump": [1],
        "2,1,top-bottom,spread": [0],
        "2,1,top-bottom,spread_mod": [0],
        "2,2,bottom-top,clump": [0, 0],
        "2,2,bottom-top,spread": [0, 1],
        "2,2,bottom-top,spread_mod": [0, 0],
        "2,2,random,clump": [0, 0],
        "2,2,random,spread": [1, 0],
        "2,2,random,spread_mod": [0, 0],
        "2,2,top-bottom,clump": [1, 1],
        "2,2,top-bottom,spread": [1, 0],
        "2,2,top-bottom,spread_mod": [0, 0],
        "2,3,bottom-top,clump": [0, 0, 0],
        "2,3,bottom-top,spread": [0, 0, 1],
        "2,3,bottom-top,spread_mod": [0, 0, 0],
        "2,3,random,clump": [0, 0, 0],
        "2,3,random,spread": [1, 0, 0],
        "2,3,random,spread_mod": [0, 0, 0],
        "2,3,top-bottom,clump": [1, 1, 1],
        "2,3,top-bottom,spread": [1, 1, 0],
        "2,3,top-bottom,spread_mod": [0, 0, 0],
        "3,1,bottom-top,clump": [0],
        "3,1,bottom-top,spread": [0],
        "3,1,bottom-top,spread_mod": [0],
        "3,1,random,clump": [0],
        "3,1,random,spread": [1],
        "3,1,random,spread_mod": [0],
        "3,1,top-bottom,clump": [2],
        "3,1,top-bottom,spread": [1],
        "3,1,top-bottom,spread_mod": [0],
        "3,2,bottom-top,clump": [0, 0],
        "3,2,bottom-top,spread": [0, 1],
        "3,2,bottom-top,spread_mod": [0, 1],
        "3,2,random,clump": [0, 0],
        "3,2,random,spread": [1, 0],
        "3,2,random,spread_mod": [0, 1],
        "3,2,top-bottom,clump": [2, 2],
        "3,2,top-bottom,spread": [1, 0],
        "3,2,top-bottom,spread_mod": [0, 1],
        "3,5,bottom-top,clump": [0, 0, 0, 0, 0],
        "3,5,bottom-top,spread": [0, 0, 1, 1, 2],
        "3,5,bottom-top,spread_mod": [0, 1, 0, 1, 0],
        "3,5,random,clump": [0, 0, 0, 0, 0],
        "3,5,random,spread": [1, 2, 0, 0, 1],
        "3,5,random,spread_mod": [0, 1, 0, 1, 0],
        "3,5,top-bottom,clump": [2, 2, 2, 2, 2],
        "3,5,top-bottom,spread": [2, 2, 1, 1, 0],
        "3,5,top-bottom,spread_mod": [0, 1, 0, 1, 0],
        "4,2,bottom-top,clump": [0, 0],
        "4,2,bottom-top,spread": [0, 1],
        "4,2,bottom-top,spread_mod": [0, 1],
        "4,2,random,clump": [0, 0],
        "4,2,random,spread": [1, 2],
        "4,2,random,spread_mod": [0, 1],
        "4,2,top-bottom,clump": [3, 3],
        "4,2,top-bottom,spread": [2, 1],
        "4,2,top-bottom,spread_mod": [0, 1],
        "4,4,bottom-top,clump": [0, 0, 0, 0],
        "4,4,bottom-top,spread": [0, 1, 2, 3],
        "4,4,bottom-top,spread_mod": [0, 1, 2, 0],
        "4,4,random,clump": [0, 0, 0, 0],
        "4,4,random,spread": [1, 3, 2, 0],
        "4,4,random,spread_mod": [0, 1, 2, 0],
        "4,4,top-bottom,clump": [3, 3, 3, 3],
        "4,4,top-bottom,spread": [3, 2, 1, 0],
        "4,4,top-bottom,spread_mod": [0, 1, 2, 0],
        "4,6,bottom-top,clump": [0, 0, 0, 0, 0, 0],
        "4,6,bottom-top,spread": [0, 0, 1, 2, 2, 3],
        "4,6,bottom-top,spread_mod": [0, 1, 2, 0, 1, 2],
        "4,6,random,clump": [0, 0, 0, 0, 0, 0],
        "4,6,random,spread": [1, 3, 2, 0, 0, 2],
        "4,6,random,spread_mod": [0, 1, 2, 0, 1, 2],
        "4,6,top-bottom,clump": [3, 3, 3, 3, 3, 3],
        "4,6,top-bottom,spread": [3, 3, 2, 1, 1, 0],
        "4,6,top-bottom,spread_mod": [0, 1, 2, 0, 1, 2],
        "6,3,bottom-top,clump": [0, 0, 0],
        "6,3,bottom-top,spread": [0, 1, 3],
        "6,3,bottom-top,spread_mod": [0, 1, 2],
        "6,3,random,clump": [4, 4, 4],
        "6,3,random,spread": [0, 2, 4],
        "6,3,random,spread_mod": [0, 1, 2],
        "6,3,top-bottom,clump": [5, 5, 5],
        "6,3,top-bottom,spread": [4, 3, 1],
        "6,3,top-bottom,spread_mod": [0, 1, 2],
        "6,8,bottom-top,clump": [0, 0, 0, 0, 0, 0, 0, 0],
        "6,8,bottom-top,spread": [0, 0, 1, 2, 3, 3, 4, 5],
        "6,8,bottom-top,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "6,8,random,clump": [4, 4, 4, 4, 4, 4, 4, 4],
        "6,8,random,spread": [4, 0, 2, 5, 3, 1, 1, 3],
        "6,8,random,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "6,8,top-bottom,clump": [5, 5, 5, 5, 5, 5, 5, 5],
        "6,8,top-bottom,spread": [5, 5, 4, 3, 2, 2, 1, 0],
        "6,8,top-bottom,spread_mod": [0, 1, 2, 3, 4, 0, 1, 2],
        "8,3,bottom-top,clump": [0, 0, 0],
        "8,3,bottom-top,spread": [0, 2, 4],
        "8,3,bottom-top,spread_mod": [0, 1, 2],
        "8,3,random,clump": [4, 4, 4],
        "8,3,random,spread": [4, 0, 2],
        "8,3,random,spread_mod": [0, 1, 2],
        "8,3,top-bottom,clump": [7, 7, 7],
        "8,3,top-bottom,spread": [6, 4, 2],
        "8,3,top-bottom,spread_mod": [0, 1, 2],
    },
    "locate_collapse_pos:decoder": {
        "all_but_last,-1": "ERR ValueError",
        "all_but_last,0": 0,
        "all_but_last,1": 0,
        "all_but_last,2": 0,
        "all_but_last,3": 1,
        "all_but_last,4": "ERR ValueError",
        "head_merged,-1": "ERR ValueError",
        "head_merged,0": 0,
        "head_merged,1": 0,
        "head_merged,2": 1,
        "head_merged,3": 2,
        "head_merged,4": "ERR ValueError",
        "mid_merged,-1": "ERR ValueError",
        "mid_merged,0": 0,
        "mid_merged,1": 1,
        "mid_merged,2": 1,
        "mid_merged,3": 2,
        "mid_merged,4": "ERR ValueError",
        "singletons4,-1": "ERR ValueError",
        "singletons4,0": 0,
        "singletons4,1": 1,
        "singletons4,2": 2,
        "singletons4,3": 3,
        "singletons4,4": "ERR ValueError",
        "tail_merged,-1": "ERR ValueError",
        "tail_merged,0": 0,
        "tail_merged,1": 1,
        "tail_merged,2": 2,
        "tail_merged,3": 2,
        "tail_merged,4": "ERR ValueError",
    },
    "locate_collapse_pos:vision": {
        "all_but_last,-1": "ERR ValueError",
        "all_but_last,0": 0,
        "all_but_last,1": 0,
        "all_but_last,2": 0,
        "all_but_last,3": 0,
        "all_but_last,4": "ERR ValueError",
        "head_merged,-1": "ERR ValueError",
        "head_merged,0": 0,
        "head_merged,1": 0,
        "head_merged,2": 1,
        "head_merged,3": 1,
        "head_merged,4": "ERR ValueError",
        "mid_merged,-1": "ERR ValueError",
        "mid_merged,0": 0,
        "mid_merged,1": 1,
        "mid_merged,2": 1,
        "mid_merged,3": 1,
        "mid_merged,4": "ERR ValueError",
        "singletons4,-1": "ERR ValueError",
        "singletons4,0": 0,
        "singletons4,1": 1,
        "singletons4,2": 2,
        "singletons4,3": 2,
        "singletons4,4": "ERR ValueError",
        "tail_merged,-1": "ERR ValueError",
        "tail_merged,0": 0,
        "tail_merged,1": 1,
        "tail_merged,2": 1,
        "tail_merged,3": 1,
        "tail_merged,4": "ERR ValueError",
    },
    "plan_inserted_positions": {
        "1,1,bottom-top,clump": {"positions": [1], "schedule": [0]},
        "1,1,bottom-top,spread": {"positions": [1], "schedule": [0]},
        "1,1,bottom-top,spread_mod": "ERR ZeroDivisionError",
        "1,1,top-bottom,clump": {"positions": [1], "schedule": [0]},
        "1,1,top-bottom,spread": {"positions": [1], "schedule": [0]},
        "1,1,top-bottom,spread_mod": "ERR ZeroDivisionError",
        "2,1,bottom-top,clump": {"positions": [1], "schedule": [0]},
        "2,1,bottom-top,spread": {"positions": [1], "schedule": [0]},
        "2,1,bottom-top,spread_mod": {"positions": [1], "schedule": [0]},
        "2,1,top-bottom,clump": {"positions": [2], "schedule": [1]},
        "2,1,top-bottom,spread": {"positions": [1], "schedule": [0]},
        "2,1,top-bottom,spread_mod": {"positions": [1], "schedule": [0]},
        "2,2,bottom-top,clump": {"positions": [1, 2], "schedule": [0, 0]},
        "2,2,bottom-top,spread": {"positions": [1, 3], "schedule": [0, 1]},
        "2,2,bottom-top,spread_mod": {"positions": [1, 2], "schedule": [0, 0]},
        "2,2,top-bottom,clump": {"positions": [2, 3], "schedule": [1, 1]},
        "2,2,top-bottom,spread": {"positions": [3, 1], "schedule": [1, 0]},
        "2,2,top-bottom,spread_mod": {"positions": [1, 2], "schedule": [0, 0]},
        "2,3,bottom-top,clump": {"positions": [1, 2, 3], "schedule": [0, 0, 0]},
        "2,3,bottom-top,spread": {"positions": [1, 2, 4], "schedule": [0, 0, 1]},
        "2,3,bottom-top,spread_mod": {"positions": [1, 2, 3], "schedule": [0, 0, 0]},
        "2,3,top-bottom,clump": {"positions": [2, 3, 4], "schedule": [1, 1, 1]},
        "2,3,top-bottom,spread": {"positions": [3, 4, 1], "schedule": [1, 1, 0]},
        "2,3,top-bottom,spread_mod": {"positions": [1, 2, 3], "schedule": [0, 0, 0]},
        "3,1,bottom-top,clump": {"positions": [1], "schedule": [0]},
        "3,1,bottom-top,spread": {"positions": [1], "schedule": [0]},
        "3,1,bottom-top,spread_mod": {"positions": [1], "schedule": [0]},
        "3,1,top-bottom,clump": {"positions": [3], "schedule": [2]},
        "3,1,top-bottom,spread": {"positions": [2], "schedule": [1]},
        "3,1,top-bottom,spread_mod": {"positions": [1], "schedule": [0]},
        "3,2,bottom-top,clump": {"positions": [1, 2], "schedule": [0, 0]},
        "3,2,bottom-top,spread": {"positions": [1, 3], "schedule": [0, 1]},
        "3,2,bottom-top,spread_mod": {"positions": [1, 3], "schedule": [0, 1]},
        "3,2,top-bottom,clump": {"positions": [3, 4], "schedule": [2, 2]},
        "3,2,top-bottom,spread": {"positions": [3, 1], "schedule": [1, 0]},
        "3,2,top-bottom,spread_mod": {"positions": [1, 3], "schedule": [0, 1]},
        "3,5,bottom-top,clump": {"positions": [1, 2, 3, 4, 5], "schedule": [0, 0, 0, 0, 0]},
        "3,5,bottom-top,spread": {"positions": [1, 2, 4, 5, 7], "schedule": [0, 0, 1, 1, 2]},
        "3,5,bottom-top,spread_mod": {"positions": [1, 5, 2, 6, 3], "schedule": [0, 1, 0, 1, 0]},
        "3,5,top-bottom,clump": {"positions": [3, 4, 5, 6, 7], "schedule": [2, 2, 2, 2, 2]},
        "3,5,top-bottom,spread": {"positions": [6, 7, 3, 4, 1], "schedule": [2, 2, 1, 1, 0]},
        "3,5,top-bottom,spread_mod": {"positions": [1, 5, 2, 6, 3], "schedule": [0, 1, 0, 1, 0]},
        "4,2,bottom-top,clump": {"positions": [1, 2], "schedule": [0, 0]},
        "4,2,bottom-top,spread": {"positions": [1, 3], "schedule": [0, 1]},
        "4,2,bottom-top,spread_mod": {"positions": [1, 3], "schedule": [0, 1]},
        "4,2,top-bottom,clump": {"positions": [4, 5], "schedule": [3, 3]},
        "4,2,top-bottom,spread": {"positions": [4, 2], "schedule": [2, 1]},
        "4,2,top-bottom,spread_mod": {"positions": [1, 3], "schedule": [0, 1]},
        "4,4,bottom-top,clump": {"positions": [1, 2, 3, 4], "schedule": [0, 0, 0, 0]},
        "4,4,bottom-top,spread": {"positions": [1, 3, 5, 7], "schedule": [0, 1, 2, 3]},
        "4,4,bottom-top,spread_mod": {"positions": [1, 4, 6, 2], "schedule": [0, 1, 2, 0]},
        "4,4,top-bottom,clump": {"positions": [4, 5, 6, 7], "schedule": [3, 3, 3, 3]},
        "4,4,top-bottom,spread": {"positions": [7, 5, 3, 1], "schedule": [3, 2, 1, 0]},
        "4,4,top-bottom,spread_mod": {"positions": [1, 4, 6, 2], "schedule": [0, 1, 2, 0]},
        "4,6,bottom-top,clump": {"positions": [1, 2, 3, 4, 5, 6], "schedule": [0, 0, 0, 0, 0, 0]},
        "4,6,bottom-top,spread": {"positions": [1, 2, 4, 6, 7, 9], "schedule": [0, 0, 1, 2, 2, 3]},
        "4,6,bottom-top,spread_mod": {"positions": [1, 4, 7, 2, 5, 8], "schedule": [0, 1, 2, 0, 1, 2]},
        "4,6,top-bottom,clump": {"positions": [4, 5, 6, 7, 8, 9], "schedule": [3, 3, 3, 3, 3, 3]},
        "4,6,top-bottom,spread": {"positions": [8, 9, 6, 3, 4, 1], "schedule": [3, 3, 2, 1, 1, 0]},
        "4,6,top-bottom,spread_mod": {"positions": [1, 4, 7, 2, 5, 8], "schedule": [0, 1, 2, 0, 1, 2]},
        "6,3,bottom-top,clump": {"positions": [1, 2, 3], "schedule": [0, 0, 0]},
        "6,3,bottom-top,spread": {"positions": [1, 3, 6], "schedule": [0, 1, 3]},
        "6,3,bottom-top,spread_mod": {"positions": [1, 3, 5], "schedule": [0, 1, 2]},
        "6,3,top-bottom,clump": {"positions": [6, 7, 8], "schedule": [5, 5, 5]},
        "6,3,top-bottom,spread": {"positions": [7, 5, 2], "schedule": [4, 3, 1]},
        "6,3,top-bottom,spread_mod": {"positions": [1, 3, 5], "schedule": [0, 1, 2]},
        "6,8,bottom-top,clump": {"positions": [1, 2, 3, 4, 5, 6, 7, 8], "schedule": [0, 0, 0, 0, 0, 0, 0, 0]},
        "6,8,bottom-top,spread": {"positions": [1, 2, 4, 6, 8, 9, 11, 13], "schedule": [0, 0, 1, 2, 3, 3, 4, 5]},
        "6,8,bottom-top,spread_mod": {"positions": [1, 4, 7, 10, 12, 2, 5, 8], "schedule": [0, 1, 2, 3, 4, 0, 1, 2]},
        "6,8,top-bottom,clump": {"positions": [6, 7, 8, 9, 10, 11, 12, 13], "schedule": [5, 5, 5, 5, 5, 5, 5, 5]},
        "6,8,top-bottom,spread": {"positions": [12, 13, 10, 8, 5, 6, 3, 1], "schedule": [5, 5, 4, 3, 2, 2, 1, 0]},
        "6,8,top-bottom,spread_mod": {"positions": [1, 4, 7, 10, 12, 2, 5, 8], "schedule": [0, 1, 2, 3, 4, 0, 1, 2]},
        "8,3,bottom-top,clump": {"positions": [1, 2, 3], "schedule": [0, 0, 0]},
        "8,3,bottom-top,spread": {"positions": [1, 4, 7], "schedule": [0, 2, 4]},
        "8,3,bottom-top,spread_mod": {"positions": [1, 3, 5], "schedule": [0, 1, 2]},
        "8,3,top-bottom,clump": {"positions": [8, 9, 10], "schedule": [7, 7, 7]},
        "8,3,top-bottom,spread": {"positions": [9, 6, 3], "schedule": [6, 4, 2]},
        "8,3,top-bottom,spread_mod": {"positions": [1, 3, 5], "schedule": [0, 1, 2]},
    },
    "realized_spans:decoder": {
        "1,1,bottom-top,clump": "ERR ValueError",
        "1,1,bottom-top,spread": "ERR ValueError",
        "1,1,bottom-top,spread_mod": "ERR ValueError",
        "1,1,random,clump": "ERR ValueError",
        "1,1,random,spread": "ERR ValueError",
        "1,1,random,spread_mod": "ERR ValueError",
        "1,1,top-bottom,clump": "ERR ValueError",
        "1,1,top-bottom,spread": "ERR ValueError",
        "1,1,top-bottom,spread_mod": "ERR ValueError",
        "2,1,bottom-top,clump": [(0, 1)],
        "2,1,bottom-top,spread": [(0, 1)],
        "2,1,bottom-top,spread_mod": [(0, 1)],
        "2,1,random,clump": [(0, 1)],
        "2,1,random,spread": [(0, 1)],
        "2,1,random,spread_mod": [(0, 1)],
        "2,1,top-bottom,clump": [(0, 1)],
        "2,1,top-bottom,spread": [(0, 1)],
        "2,1,top-bottom,spread_mod": [(0, 1)],
        "3,1,bottom-top,clump": [(0, 1), (2,)],
        "3,1,bottom-top,spread": [(0, 1), (2,)],
        "3,1,bottom-top,spread_mod": [(0, 1), (2,)],
        "3,1,random,clump": [(0,), (1, 2)],
        "3,1,random,spread": [(0, 1), (2,)],
        "3,1,random,spread_mod": [(0, 1), (2,)],
        "3,1,top-bottom,clump": [(0,), (1, 2)],
        "3,1,top-bottom,spread": [(0,), (1, 2)],
        "3,1,top-bottom,spread_mod": [(0, 1), (2,)],
        "3,2,bottom-top,clump": [(0, 1, 2)],
        "3,2,bottom-top,spread": [(0, 1, 2)],
        "3,2,bottom-top,spread_mod": [(0, 1, 2)],
        "3,2,random,clump": [(0, 1, 2)],
        "3,2,random,spread": [(0, 1, 2)],
        "3,2,random,spread_mod": [(0, 1, 2)],
        "3,2,top-bottom,clump": "ERR IndexError",
        "3,2,top-bottom,spread": [(0, 1, 2)],
        "3,2,top-bottom,spread_mod": [(0, 1, 2)],
        "4,1,bottom-top,clump": [(0, 1), (2,), (3,)],
        "4,1,bottom-top,spread": [(0, 1), (2,), (3,)],
        "4,1,bottom-top,spread_mod": [(0, 1), (2,), (3,)],
        "4,1,random,clump": [(0,), (1,), (2, 3)],
        "4,1,random,spread": [(0, 1), (2,), (3,)],
        "4,1,random,spread_mod": [(0, 1), (2,), (3,)],
        "4,1,top-bottom,clump": [(0,), (1,), (2, 3)],
        "4,1,top-bottom,spread": [(0,), (1,), (2, 3)],
        "4,1,top-bottom,spread_mod": [(0, 1), (2,), (3,)],
        "4,2,bottom-top,clump": [(0, 1, 2), (3,)],
        "4,2,bottom-top,spread": [(0, 1, 2), (3,)],
        "4,2,bottom-top,spread_mod": [(0, 1, 2), (3,)],
        "4,2,random,clump": [(0,), (1, 2, 3)],
        "4,2,random,spread": [(0, 1, 2), (3,)],
        "4,2,random,spread_mod": [(0, 1, 2), (3,)],
        "4,2,top-bottom,clump": "ERR IndexError",
        "4,2,top-bottom,spread": [(0,), (1, 2, 3)],
        "4,2,top-bottom,spread_mod": [(0, 1, 2), (3,)],
        "4,3,bottom-top,clump": [(0, 1, 2, 3)],
        "4,3,bottom-top,spread": [(0, 1, 2, 3)],
        "4,3,bottom-top,spread_mod": [(0, 1, 2, 3)],
        "4,3,random,clump": [(0, 1, 2, 3)],
        "4,3,random,spread": [(0, 1, 2, 3)],
        "4,3,random,spread_mod": [(0, 1, 2, 3)],
        "4,3,top-bottom,clump": "ERR IndexError",
        "4,3,top-bottom,spread": [(0, 1, 2, 3)],
        "4,3,top-bottom,spread_mod": [(0, 1, 2, 3)],
        "6,2,bottom-top,clump": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,bottom-top,spread": [(0, 1), (2, 3), (4,), (5,)],
        "6,2,bottom-top,spread_mod": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,random,clump": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,random,spread": [(0,), (1,), (2,), (3, 4, 5)],
        "6,2,random,spread_mod": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,top-bottom,clump": "ERR IndexError",
        "6,2,top-bottom,spread": [(0,), (1,), (2, 3), (4, 5)],
        "6,2,top-bottom,spread_mod": [(0, 1, 2), (3,), (4,), (5,)],
        "6,3,bottom-top,clump": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,bottom-top,spread": [(0, 1, 2), (3, 4), (5,)],
        "6,3,bottom-top,spread_mod": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,random,clump": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,random,spread": [(0,), (1,), (2, 3, 4, 5)],
        "6,3,random,spread_mod": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,top-bottom,clump": "ERR IndexError",
        "6,3,top-bottom,spread": [(0,), (1, 2), (3, 4, 5)],
        "6,3,top-bottom,spread_mod": [(0, 1, 2, 3), (4,), (5,)],
        "6,5,bottom-top,clump": [(0, 1, 2, 3, 4, 5)],
        "6,5,bottom-top,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,bottom-top,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,clump": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "6,5,top-bottom,clump": "ERR IndexError",
        "6,5,top-bottom,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,top-bottom,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "8,3,bottom-top,clump": [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
        "8,3,bottom-top,spread": [(0, 1), (2, 3), (4, 5), (6,), (7,)],
        "8,3,bottom-top,spread_mod": [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
        "8,3,random,clump": [(0, 1, 2), (3,), (4,), (5,), (6, 7)],
        "8,3,random,spread": [(0,), (1,), (2,), (3, 4, 5, 6), (7,)],
        "8,3,random,spread_mod": [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
        "8,3,top-bottom,clump": "ERR IndexError",
        "8,3,top-bottom,spread": [(0,), (1,), (2, 3), (4, 5), (6, 7)],
        "8,3,top-bottom,spread_mod": [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
        "8,4,bottom-top,clump": [(0, 1, 2, 3, 4), (5,), (6,), (7,)],
        "8,4,bottom-top,spread": [(0, 1, 2), (3, 4), (5, 6), (7,)],
        "8,4,bottom-top,spread_mod": [(0, 1, 2, 3, 4), (5,), (6,), (7,)],
        "8,4,random,clump": [(0, 1, 2, 3), (4,), (5,), (6, 7)],
        "8,4,random,spread": [(0,), (1,), (2, 3, 4, 5, 6), (7,)],
        "8,4,random,spread_mod": [(0, 1, 2, 3, 4), (5,), (6,), (7,)],
        "8,4,top-bottom,clump": "ERR IndexError",
        "8,4,top-bottom,spread": [(0,), (1, 2), (3, 4), (5, 6, 7)],
        "8,4,top-bottom,spread_mod": [(0, 1, 2, 3, 4), (5,), (6,), (7,)],
    },
    "realized_spans:vision": {
        "1,1,bottom-top,clump": "ERR ValueError",
        "1,1,bottom-top,spread": "ERR ValueError",
        "1,1,bottom-top,spread_mod": "ERR ValueError",
        "1,1,random,clump": "ERR ValueError",
        "1,1,random,spread": "ERR ValueError",
        "1,1,random,spread_mod": "ERR ValueError",
        "1,1,top-bottom,clump": "ERR ValueError",
        "1,1,top-bottom,spread": "ERR ValueError",
        "1,1,top-bottom,spread_mod": "ERR ValueError",
        "2,1,bottom-top,clump": [(0, 1)],
        "2,1,bottom-top,spread": [(0, 1)],
        "2,1,bottom-top,spread_mod": [(0, 1)],
        "2,1,random,clump": [(0, 1)],
        "2,1,random,spread": [(0, 1)],
        "2,1,random,spread_mod": [(0, 1)],
        "2,1,top-bottom,clump": [(0, 1)],
        "2,1,top-bottom,spread": [(0, 1)],
        "2,1,top-bottom,spread_mod": [(0, 1)],
        "3,1,bottom-top,clump": [(0, 1), (2,)],
        "3,1,bottom-top,spread": [(0, 1), (2,)],
        "3,1,bottom-top,spread_mod": [(0, 1), (2,)],
        "3,1,random,clump": [(0,), (1, 2)],
        "3,1,random,spread": [(0, 1), (2,)],
        "3,1,random,spread_mod": [(0, 1), (2,)],
        "3,1,top-bottom,clump": [(0,), (1, 2)],
        "3,1,top-bottom,spread": [(0,), (1, 2)],
        "3,1,top-bottom,spread_mod": [(0,), (1, 2)],
        "3,2,bottom-top,clump": [(0, 1, 2)],
        "3,2,bottom-top,spread": [(0, 1, 2)],
        "3,2,bottom-top,spread_mod": [(0, 1, 2)],
        "3,2,random,clump": [(0, 1, 2)],
        "3,2,random,spread": [(0, 1, 2)],
        "3,2,random,spread_mod": [(0, 1, 2)],
        "3,2,top-bottom,clump": [(0, 1, 2)],
        "3,2,top-bottom,spread": [(0, 1, 2)],
        "3,2,top-bottom,spread_mod": [(0, 1, 2)],
        "4,1,bottom-top,clump": [(0, 1), (2,), (3,)],
        "4,1,bottom-top,spread": [(0, 1), (2,), (3,)],
        "4,1,bottom-top,spread_mod": [(0, 1), (2,), (3,)],
        "4,1,random,clump": [(0,), (1,), (2, 3)],
        "4,1,random,spread": [(0, 1), (2,), (3,)],
        "4,1,random,spread_mod": [(0, 1), (2,), (3,)],
        "4,1,top-bottom,clump": [(0,), (1,), (2, 3)],
        "4,1,top-bottom,spread": [(0,), (1,), (2, 3)],
        "4,1,top-bottom,spread_mod": [(0,), (1,), (2, 3)],
        "4,2,bottom-top,clump": [(0, 1, 2), (3,)],
        "4,2,bottom-top,spread": [(0, 1, 2), (3,)],
        "4,2,bottom-top,spread_mod": [(0, 1), (2, 3)],
        "4,2,random,clump": [(0,), (1, 2, 3)],
        "4,2,random,spread": [(0, 1, 2), (3,)],
        "4,2,random,spread_mod": [(0, 1), (2, 3)],
        "4,2,top-bottom,clump": [(0,), (1, 2, 3)],
        "4,2,top-bottom,spread": [(0,), (1, 2, 3)],
        "4,2,top-bottom,spread_mod": [(0, 1), (2, 3)],
        "4,3,bottom-top,clump": [(0, 1, 2, 3)],
        "4,3,bottom-top,spread": [(0, 1, 2, 3)],
        "4,3,bottom-top,spread_mod": [(0, 1, 2, 3)],
        "4,3,random,clump": [(0, 1, 2, 3)],
        "4,3,random,spread": [(0, 1, 2, 3)],
        "4,3,random,spread_mod": [(0, 1, 2, 3)],
        "4,3,top-bottom,clump": [(0, 1, 2, 3)],
        "4,3,top-bottom,spread": [(0, 1, 2, 3)],
        "4,3,top-bottom,spread_mod": [(0, 1, 2, 3)],
        "6,2,bottom-top,clump": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,bottom-top,spread": [(0, 1), (2, 3), (4,), (5,)],
        "6,2,bottom-top,spread_mod": [(0, 1), (2,), (3,), (4, 5)],
        "6,2,random,clump": [(0, 1, 2), (3,), (4,), (5,)],
        "6,2,random,spread": [(0,), (1,), (2,), (3, 4, 5)],
        "6,2,random,spread_mod": [(0, 1), (2,), (3,), (4, 5)],
        "6,2,top-bottom,clump": [(0,), (1,), (2,), (3, 4, 5)],
        "6,2,top-bottom,spread": [(0,), (1,), (2, 3), (4, 5)],
        "6,2,top-bottom,spread_mod": [(0, 1), (2,), (3,), (4, 5)],
        "6,3,bottom-top,clump": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,bottom-top,spread": [(0, 1, 2), (3, 4), (5,)],
        "6,3,bottom-top,spread_mod": [(0, 1), (2, 3), (4, 5)],
        "6,3,random,clump": [(0, 1, 2, 3), (4,), (5,)],
        "6,3,random,spread": [(0,), (1,), (2, 3, 4, 5)],
        "6,3,random,spread_mod": [(0, 1), (2, 3), (4, 5)],
        "6,3,top-bottom,clump": [(0,), (1,), (2, 3, 4, 5)],
        "6,3,top-bottom,spread": [(0,), (1, 2), (3, 4, 5)],
        "6,3,top-bottom,spread_mod": [(0, 1), (2, 3), (4, 5)],
        "6,5,bottom-top,clump": [(0, 1, 2, 3, 4, 5)],
        "6,5,bottom-top,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,bottom-top,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,clump": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,random,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "6,5,top-bottom,clump": [(0, 1, 2, 3, 4, 5)],
        "6,5,top-bottom,spread": [(0, 1, 2, 3, 4, 5)],
        "6,5,top-bottom,spread_mod": [(0, 1, 2, 3, 4, 5)],
        "8,3,bottom-top,clump": [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
        "8,3,bottom-top,spread": [(0, 1), (2, 3), (4, 5), (6,), (7,)],
        "8,3,bottom-top,spread_mod": [(0, 1), (2,), (3, 4), (5,), (6, 7)],
        "8,3,random,clump": [(0, 1, 2), (3,), (4,), (5,), (6, 7)],
        "8,3,random,spread": [(0,), (1,), (2,), (3, 4, 5, 6), (7,)],
        "8,3,random,spread_mod": [(0, 1), (2,), (3, 4), (5,), (6, 7)],
        "8,3,top-bottom,clump": [(0,), (1,), (2,), (3,), (4, 5, 6, 7)],
        "8,3,top-bottom,spread": [(0,), (1,), (2, 3), (4, 5), (6, 7)],
        "8,3,top-bottom,spread_mod": [(0, 1), (2,), (3, 4), (5,), (6, 7)],
        "8,4,bottom-top,clump": [(0, 1, 2, 3, 4), (5,), (6,), (7,)],
        "8,4,bottom-top,spread": [(0, 1, 2), (3, 4), (5, 6), (7,)],
        "8,4,bottom-top,spread_mod": [(0, 1), (2, 3), (4, 5), (6, 7)],
        "8,4,random,clump": [(0, 1, 2, 3), (4,), (5,), (6, 7)],
        "8,4,random,spread": [(0,), (1,), (2, 3, 4, 5, 6), (7,)],
        "8,4,random,spread_mod": [(0, 1), (2, 3), (4, 5), (6, 7)],
        "8,4,top-bottom,clump": [(0,), (1,), (2,), (3, 4, 5, 6, 7)],
        "8,4,top-bottom,spread": [(0,), (1, 2), (3, 4), (5, 6, 7)],
        "8,4,top-bottom,spread_mod": [(0, 1), (2, 3), (4, 5), (6, 7)],
    },
    "spread_anchor_schedule": {
        "0,0,bogus": [],
        "0,0,bottom-top": [],
        "0,0,random": [],
        "0,0,top-bottom": [],
        "0,1,bogus": [],
        "0,1,bottom-top": [],
        "0,1,random": [],
        "0,1,top-bottom": [],
        "0,2,bogus": [],
        "0,2,bottom-top": [],
        "0,2,random": [],
        "0,2,top-bottom": [],
        "0,3,bogus": [],
        "0,3,bottom-top": [],
        "0,3,random": [],
        "0,3,top-bottom": [],
        "0,5,bogus": [],
        "0,5,bottom-top": [],
        "0,5,random": [],
        "0,5,top-bottom": [],
        "0,8,bogus": [],
        "0,8,bottom-top": [],
        "0,8,random": [],
        "0,8,top-bottom": [],
        "1,0,bogus": [],
        "1,0,bottom-top": [],
        "1,0,random": [],
        "1,0,top-bottom": [],
        "1,1,bogus": "ERR ValueError",
        "1,1,bottom-top": [0],
        "1,1,random": [0],
        "1,1,top-bottom": [0],
        "1,2,bogus": "ERR ValueError",
        "1,2,bottom-top": [0],
        "1,2,random": [0],
        "1,2,top-bottom": [1],
        "1,3,bogus": "ERR ValueError",
        "1,3,bottom-top": [0],
        "1,3,random": [0],
        "1,3,top-bottom": [2],
        "1,5,bogus": "ERR ValueError",
        "1,5,bottom-top": [0],
        "1,5,random": [4],
        "1,5,top-bottom": [4],
        "1,8,bogus": "ERR ValueError",
        "1,8,bottom-top": [0],
        "1,8,random": [4],
        "1,8,top-bottom": [7],
        "2,0,bogus": [],
        "2,0,bottom-top": [],
        "2,0,random": [],
        "2,0,top-bottom": [],
        "2,1,bogus": "ERR ValueError",
        "2,1,bottom-top": [0, 0],
        "2,1,random": [0, 0],
        "2,1,top-bottom": [0, 0],
        "2,2,bogus": "ERR ValueError",
        "2,2,bottom-top": [0, 1],
        "2,2,random": [0, 1],
        "2,2,top-bottom": [1, 0],
        "2,3,bogus": "ERR ValueError",
        "2,3,bottom-top": [0, 1],
        "2,3,random": [0, 1],
        "2,3,top-bottom": [2, 1],
        "2,5,bogus": "ERR ValueError",
        "2,5,bottom-top": [0, 2],
        "2,5,random": [4, 3],
        "2,5,top-bottom": [4, 2],
        "2,8,bogus": "ERR ValueError",
        "2,8,bottom-top": [0, 4],
        "2,8,random": [4, 5],
        "2,8,top-bottom": [7, 3],
        "3,0,bogus": [],
        "3,0,bottom-top": [],
        "3,0,random": [],
        "3,0,top-bottom": [],
        "3,1,bogus": "ERR ValueError",
        "3,1,bottom-top": [0, 0, 0],
        "3,1,random": [0, 0, 0],
        "3,1,top-bottom": [0, 0, 0],
        "3,2,bogus": "ERR ValueError",
        "3,2,bottom-top": [0, 0, 1],
        "3,2,random": [0, 1, 1],
        "3,2,top-bottom": [1, 1, 0],
        "3,3,bogus": "ERR ValueError",
        "3,3,bottom-top": [0, 1, 2],
        "3,3,random": [0, 1, 2],
        "3,3,top-bottom": [2, 1, 0],
        "3,5,bogus": "ERR ValueError",
        "3,5,bottom-top": [0, 1, 3],
        "3,5,random": [4, 3, 2],
        "3,5,top-bottom": [4, 3, 1],
        "3,8,bogus": "ERR ValueError",
        "3,8,bottom-top": [0, 2, 5],
        "3,8,random": [4, 5, 3],
        "3,8,top-bottom": [7, 5, 2],
        "5,0,bogus": [],
        "5,0,bottom-top": [],
        "5,0,random": [],
        "5,0,top-bottom": [],
        "5,1,bogus": "ERR ValueError",
        "5,1,bottom-top": [0, 0, 0, 0, 0],
        "5,1,random": [0, 0, 0, 0, 0],
        "5,1,top-bottom": [0, 0, 0, 0, 0],
        "5,2,bogus": "ERR ValueError",
        "5,2,bottom-top": [0, 0, 0, 1, 1],
        "5,2,random": [0, 1, 1, 0, 0],
        "5,2,top-bottom": [1, 1, 1, 0, 0],
        "5,3,bogus": "ERR ValueError",
        "5,3,bottom-top": [0, 0, 1, 1, 2],
        "5,3,random": [0, 1, 2, 1, 2],
        "5,3,top-bottom": [2, 2, 1, 1, 0],
        "5,5,bogus": "ERR ValueError",
        "5,5,bottom-top": [0, 1, 2, 3, 4],
        "5,5,random": [4, 3, 2, 0, 1],
        "5,5,top-bottom": [4, 3, 2, 1, 0],
        "5,8,bogus": "ERR ValueError",
        "5,8,bottom-top": [0, 1, 3, 4, 6],
        "5,8,random": [4, 5, 3, 2, 0],
        "5,8,top-bottom": [7, 6, 4, 3, 1],
        "7,0,bogus": [],
        "7,0,bottom-top": [],
        "7,0,random": [],
        "7,0,top-bottom": [],
        "7,1,bogus": "ERR ValueError",
        "7,1,bottom-top": [0, 0, 0, 0, 0, 0, 0],
        "7,1,random": [0, 0, 0, 0, 0, 0, 0],
        "7,1,top-bottom": [0, 0, 0, 0, 0, 0, 0],
        "7,2,bogus": "ERR ValueError",
        "7,2,bottom-top": [0, 0, 0, 0, 1, 1, 1],
        "7,2,random": [0, 1, 1, 0, 0, 1, 1],
        "7,2,top-bottom": [1, 1, 1, 1, 0, 0, 0],
        "7,3,bogus": "ERR ValueError",
        "7,3,bottom-top": [0, 0, 0, 1, 1, 2, 2],
        "7,3,random": [0, 1, 2, 1, 2, 0, 0],
        "7,3,top-bottom": [2, 2, 2, 1, 1, 0, 0],
        "7,5,bogus": "ERR ValueError",
        "7,5,bottom-top": [0, 0, 1, 2, 2, 3, 4],
        "7,5,random": [4, 3, 2, 0, 1, 0, 2],
        "7,5,top-bottom": [4, 4, 3, 2, 2, 1, 0],
        "7,8,bogus": "ERR ValueError",
        "7,8,bottom-top": [0, 1, 2, 3, 4, 5, 6],
        "7,8,random": [4, 5, 3, 2, 0, 1, 6],
        "7,8,top-bottom": [7, 6, 5, 4, 3, 2, 1],
    },
}

if __name__ == "__main__":  # prints the TABLES literal for splicing into this file
    _tables = _all_tables()
    print("TABLES: dict = " + pprint.pformat(_tables, width=110, sort_dicts=True))
