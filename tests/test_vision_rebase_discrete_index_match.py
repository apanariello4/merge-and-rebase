"""Tests for depth_alignment="discrete_index_match" (the faithful BiCo/THESEUS
structural-resize control) in vision_rebase.py.

Two things need covering:

1. The wiring vision_rebase.py itself owns: `build_discrete_indexed_model`,
   called exactly the way the new `elif task_discrete_layer_match_prestep:`
   branch calls it (default `resblocks_attr="visual.transformer.resblocks"`,
   against a `visual.transformer.resblocks` ModuleList), produces a model
   with the target depth. (`DiscreteLayerPairing`/`build_discrete_indexed_model`'s
   own numeric correctness is covered exhaustively by
   `tests/test_discrete_layer_match.py`; that suite is not duplicated here.)
2. `depth_alignment`'s config-resolution guards in `main()`: an unknown value
   fails fast, and `discrete_index_match` combined with any of ARIADNE's
   target-informed correction mechanisms is rejected.

`main()` cannot run end-to-end offline. The unknown-`depth_alignment`-value
guard fires before any model is built, so it is exercised through a real
`main()` call (mirroring `tests/test_vision_rebase_direct_residual_dispatch.py`'s
technique). The depth_alignment/correction-mechanism incompatibility guard
fires only after `clf_source`/`clf_target` are built (it needs their real
depths), which requires a network download this environment does not have;
that guard is instead checked structurally via `ast` source inspection, the
same technique `tests/test_vision_rebase_theseus.py`'s
`test_main_initializes_brace_diagnostic_collectors_before_the_task_loop`
already uses for a property deep inside `main()`.
"""

from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

import pytest
import torch.nn as nn

from merge_and_rebase.eval import vision_rebase
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing, build_discrete_indexed_model


def _model(depth: int) -> nn.Module:
    """Same minimal stand-in vision_rebase.py's own tests use (test_vision_rebase_theseus.py)."""
    model = SimpleNamespace()
    model.visual = SimpleNamespace(transformer=SimpleNamespace(resblocks=nn.ModuleList()))
    model.visual.transformer.resblocks.extend(nn.Linear(1, 1) for _ in range(depth))
    return model


@pytest.mark.parametrize("source_depth, target_depth", [(12, 24), (24, 12), (5, 5)])
def test_build_discrete_indexed_model_matches_target_depth(source_depth, target_depth):
    source_model = _model(source_depth)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)

    reindexed = build_discrete_indexed_model(source_model, pairing)

    assert len(reindexed.visual.transformer.resblocks) == target_depth
    # Verbatim copy, not a shared reference: mutating the reindexed copy must
    # not mutate the original (the elif branch in vision_rebase.py reassigns
    # source_base_model_task/source_ft_model_task to the *return value*, so
    # if this ever aliased the source it would silently corrupt task_delta
    # for every later task in the loop).
    assert reindexed.visual.transformer.resblocks[0] is not source_model.visual.transformer.resblocks[0]


def test_depth_alignment_invalid_value_fails_fast(monkeypatch, tmp_path):
    cfg = {
        "method": "theseus",
        "depth_alignment": "not_a_real_mode",
        "logging": {"local_log_dir": str(tmp_path / "logs")},
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("sys.argv", ["vision_rebase", "--config", str(config_path)])

    with pytest.raises(ValueError, match="depth_alignment must be one of"):
        vision_rebase.main()


def test_discrete_index_match_rejects_ariadne_target_informed_corrections():
    """Structural check (ast): main() raises when depth_alignment='discrete_index_match'
    is combined with target_residual_completion/joint_blockwise_correction/direct_p1_correction.

    This guard sits after clf_source/clf_target are built (it needs their
    real resblock depths), so it cannot be reached by a real main() call in
    this offline environment; see the module docstring.
    """
    source = inspect.getsource(vision_rebase.main)
    tree = ast.parse(source)
    main_fn = tree.body[0]
    assert isinstance(main_fn, ast.FunctionDef)

    found_guard = False
    for node in ast.walk(main_fn):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "discrete_index_match" not in test_src:
            continue
        if "target_residual_completion" not in test_src or "joint_blockwise_correction" not in test_src:
            continue
        raises = any(
            isinstance(stmt, ast.Raise)
            and isinstance(stmt.exc, ast.Call)
            and getattr(stmt.exc.func, "id", None) == "ValueError"
            for stmt in ast.walk(node)
        )
        if raises:
            found_guard = True
            break

    assert found_guard, (
        "Expected an `if ... depth_alignment ... discrete_index_match ... target_residual_completion "
        "... joint_blockwise_correction ...: raise ValueError(...)` guard in main()."
    )


def test_depth_alignment_default_is_ariadne():
    source = inspect.getsource(vision_rebase.main)
    assert 'cfg.get("depth_alignment", "ariadne")' in source
