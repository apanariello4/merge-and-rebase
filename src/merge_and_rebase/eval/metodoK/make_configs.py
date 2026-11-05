"""Seed runner for the "metodo K" protocol (docs/.local/PIANO_METODO_K.md, section 2, "Seed").

Takes one base vision config (canonical nested JSON, e.g. results/anchors_ops_20261030/configs/O2_rho0.1.json for
Procrustes/kernel + O2, or B2_bridge_ariadne_O2.json for Ariadne + O2 via the bridge) and writes one config per seed.

What varies with the seed ``s`` (the SAME integer ``s`` is used for every method, so the methods of a seed see the
same anchor pool, the same calibration draw and the same subsamples):

- ``method.params.anchor_seed``  (frame field: anchor-row draw / dedup),
- ``method.params.pool_seed``    (frame field: which pool batches paired_calibration draws),
- ``method.params.seed``         (calibration seed: E draw and the O2 G subsample; for a native Ariadne config,
                                  ``method.params.seed`` is its only seed),
- ``method.params.map_seed``     (seed of the map-rows subsamples).

What stays fixed:

- the top-level run ``seed`` (val/test split, ``eval/vision_rebase/context.py``): the alpha-selection data are the
  same in every seed, so seed variance measures the method's own randomness (anchors, calibration), not the split.
  Consequence: the test set is the same too, so the "seed + test" error of the protocol is estimated by the seed
  spread of the paired differences plus the binomial test error, not by re-splitting.

Because ``anchor_seed`` and ``pool_seed`` are frame fields (``FRAME_FIELDS`` in anchors/config.py), a new
``state_dir`` is required per seed, otherwise ``AnchorState._load_frame`` raises a mismatch. The state dir is
``<state-root>/seed<s>/<tag>`` with ``<tag>`` the basename of the base config's ``state_dir`` (encodes the frame, e.g.
``tiny_imagenet_A16_T2048``); two bases with the same tag must have the same frame fields (checked by ``--check``
across the given bases), in which case they share the state, as in a single-seed campaign. Only paths are written.

Other edits to every output config: ``logging.local_log_dir = <runs-root>/<exp>/runs/<name>_seed<s>`` (big outputs on
/data; add a symlink under results/ by hand), ``logging.run_name`` likewise, ``save`` dropped (task vectors not
needed; ``--keep-save`` keeps it, needed by metodoK_blocks for the update cosine and by depth_drift), and, with
``--full-curves`` (default), ``alpha.patience = null`` (no early stop; full val curve) and ``alpha.test_curve = true``
(test at every grid alpha; selection still uses val only).

Usage::

    python -m merge_and_rebase.eval.metodoK.make_configs --exp metodoK_noise --out-dir DIR --seeds 0 1 2 \\
        results/anchors_ops_20261030/configs/O2_rho0.1.json results/anchors_ops_20261030/configs/B2_bridge_ariadne_O2.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from merge_and_rebase.rebase.methods.anchors.config import FRAME_FIELDS

DATA_ROOT = Path("/data/merge_and_rebase_results")
DEFAULT_STATE_ROOT = DATA_ROOT / "anchors_state_metodoK"
DEFAULT_SEEDS = (0, 1, 2)
ANCHORS_SEED_KEYS = ("anchor_seed", "pool_seed", "seed", "map_seed")
FRAME_SEED_KEYS = ("anchor_seed", "pool_seed")
#: Frame fields other than the seeds, for the same-tag consistency check (imported, not mirrored, so a new frame
#: field of anchors/config.py cannot be forgotten here).
FRAME_KEYS = tuple(k for k in FRAME_FIELDS if k not in FRAME_SEED_KEYS)


def make_seed_config(
    base: dict[str, Any],
    seed: int,
    *,
    name: str,
    exp: str,
    state_root: Path = DEFAULT_STATE_ROOT,
    runs_root: Path = DATA_ROOT,
    full_curves: bool = True,
    keep_save: bool = False,
) -> dict[str, Any]:
    """One seed's config. The top-level ``seed`` (val split) is left untouched."""
    cfg = copy.deepcopy(base)
    method = cfg.get("method")
    if not isinstance(method, dict) or "name" not in method:
        raise ValueError("base config must use the canonical nested form (method: {name, params})")
    params = method.setdefault("params", {})
    if method["name"] == "anchors":
        for key in ANCHORS_SEED_KEYS:
            params[key] = int(seed)
        old = params.get("state_dir")
        tag = Path(str(old)).name if old else name
        params["state_dir"] = str(state_root / f"seed{seed}" / tag)
    elif method["name"] in ("ariadne", "direct_residual"):
        params["seed"] = int(seed)
    else:
        raise ValueError(f"unsupported method {method['name']!r} (anchors or ariadne)")
    if not keep_save:
        cfg.pop("save", None)
    if full_curves:
        alpha = cfg.setdefault("alpha", {})
        alpha["patience"] = None
        alpha["test_curve"] = True
    run_name = f"{name}_seed{seed}"
    logging = cfg.setdefault("logging", {})
    logging["local_log_dir"] = str(runs_root / exp / "runs" / run_name)
    logging["run_name"] = run_name
    return cfg


def frame_of(cfg: dict[str, Any]) -> dict[str, Any]:
    params = cfg["method"].get("params", {})
    return {k: params.get(k) for k in FRAME_KEYS}


def check_state_tags(cfgs: list[dict[str, Any]]) -> None:
    """Configs sharing a state_dir must share the frame (otherwise AnchorState raises at run time)."""
    seen: dict[str, dict[str, Any]] = {}
    for cfg in cfgs:
        sd = cfg["method"].get("params", {}).get("state_dir")
        if sd is None:
            continue
        frame = frame_of(cfg)
        if seen.setdefault(sd, frame) != frame:
            raise ValueError(f"state_dir {sd} is shared by configs with different frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bases", nargs="+", help="base config JSONs (run name = file stem)")
    ap.add_argument("--exp", required=True, help="experiment folder under /data/merge_and_rebase_results")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    ap.add_argument("--runs-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--no-full-curves", action="store_true")
    ap.add_argument("--keep-save", action="store_true")
    a = ap.parse_args()
    out: list[tuple[Path, dict[str, Any]]] = []
    for base_path in a.bases:
        base = json.loads(Path(base_path).read_text())
        name = Path(base_path).stem
        for s in a.seeds:
            cfg = make_seed_config(base, s, name=name, exp=a.exp, state_root=a.state_root, runs_root=a.runs_root,
                                   full_curves=not a.no_full_curves, keep_save=a.keep_save)
            out.append((a.out_dir / f"{name}_seed{s}.json", cfg))
    check_state_tags([c for _, c in out])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for path, cfg in out:
        path.write_text(json.dumps(cfg, indent=2))
        print(path)


if __name__ == "__main__":
    main()
