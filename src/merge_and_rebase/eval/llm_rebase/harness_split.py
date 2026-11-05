"""Fixed, versioned selection/test splits of an lm-eval task, for ``harness_split: <name>`` in an llm_rebase config.

``ifeval_k_v1`` (``ifeval_k_v1.json`` next to this file) is the IFEval split of the "metodo K" plan. It is the split
of the earlier out-of-repo script ``alpha_split_eval.py`` (L1 runs, ``split_it_anchors.json``), reproduced exactly:

1. IFEval has 541 prompts. ``calibration`` (n=40): ``order = list(range(541)); random.Random(0).shuffle(order)``;
   the first 40 of ``order`` are the calibration documents (what ``resolve_calibration_texts`` holds out with
   ``n_sequences=40, seed=0``); the other 501 are the hold-out.
2. ``torch.randperm(501, generator=manual_seed(0))`` over the sorted hold-out: the first 150 are the SELECTION
   slice (alpha is chosen here), the other 351 the TEST slice (scored on it, never used to select).

Selection, test and calibration documents are pairwise disjoint; their union is all 541 documents. The file
stores the index lists, the generator parameters and a sha256 of the lists; ``load_harness_split`` regenerates the
lists from the parameters and refuses a file that disagrees (``python -m merge_and_rebase.eval.llm_rebase.make_ifeval_split`` rewrites the file).

Config use (one key): ``harness_split: ifeval_k_v1`` fills ``harness_samples`` (selection) and
``harness_test_samples`` (test); both must then be absent from the config. Calibration must not touch the split
documents: the calibration docs of the run have to be a subset of the split's calibration docs (e.g.
``calibration_n_sequences: 40`` for IFEval), which ``check_calibration_disjoint`` enforces.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

SPLIT_DIR = Path(__file__).resolve().parent
SPLITS = {"ifeval_k_v1": SPLIT_DIR / "ifeval_k_v1.json"}
IFEVAL_K_V1_PARAMS = {"task": "ifeval", "n_docs": 541, "n_calibration": 40, "n_selection": 150, "seed": 0}


def make_split(*, task: str, n_docs: int, n_calibration: int, n_selection: int, seed: int) -> dict[str, Any]:
    """Regenerate a split from its parameters (see the module docstring); lists are sorted and disjoint."""
    order = list(range(n_docs))
    random.Random(seed).shuffle(order)  # the shuffle of llm_calibration.resolve_calibration_texts
    calibration = sorted(order[:n_calibration])
    holdout = sorted(order[n_calibration:])
    perm = torch.randperm(len(holdout), generator=torch.Generator().manual_seed(seed)).tolist()
    selection = sorted(holdout[i] for i in perm[:n_selection])
    test = sorted(holdout[i] for i in perm[n_selection:])
    return {"task": task, "n_docs": n_docs, "n_calibration": n_calibration, "n_selection": n_selection,
            "seed": seed, "selection": selection, "test": test, "calibration": calibration}


def split_sha256(split: Mapping[str, Any]) -> str:
    payload = {k: list(split[k]) for k in ("selection", "test", "calibration")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_split_file(name: str = "ifeval_k_v1") -> dict[str, Any]:
    if name != "ifeval_k_v1":
        raise ValueError(f"unknown harness split {name!r}; known: {sorted(SPLITS)}")
    data = make_split(**IFEVAL_K_V1_PARAMS)
    return {"name": name, **data, "sha256": split_sha256(data)}


def load_harness_split(name: str) -> dict[str, Any]:
    """The stored split ``{task, selection, test, calibration, ...}``, verified disjoint and regenerable."""
    if name not in SPLITS:
        raise ValueError(f"unknown harness_split {name!r}; known: {sorted(SPLITS)}")
    data = json.loads(SPLITS[name].read_text())
    params = {k: data[k] for k in ("task", "n_docs", "n_calibration", "n_selection", "seed")}
    regenerated = make_split(**params)
    for key in ("selection", "test", "calibration"):
        if [int(i) for i in data[key]] != regenerated[key]:
            raise ValueError(f"{SPLITS[name]}: stored {key!r} list does not match its generator parameters {params}")
    if data.get("sha256") != split_sha256(regenerated):
        raise ValueError(f"{SPLITS[name]}: sha256 does not match the stored lists")
    sel, tst, cal = (set(regenerated[k]) for k in ("selection", "test", "calibration"))
    if sel & tst or sel & cal or tst & cal or len(sel | tst | cal) != data["n_docs"]:
        raise ValueError(f"{SPLITS[name]}: selection / test / calibration are not a disjoint partition")
    return {**data, **{k: regenerated[k] for k in ("selection", "test", "calibration")}}


def resolve_harness_split(cfg: Mapping[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, Any]] | None:
    """``(selection samples, test samples, record)`` for ``cfg['harness_split']``; None when the key is absent/null.

    The record (name, sha256, sizes) is for the run provenance. ``harness_samples`` / ``harness_test_samples`` given
    explicitly next to ``harness_split`` are refused (two sources for one slice)."""
    name = cfg.get("harness_split", None)
    if name is None:
        return None
    clash = [k for k in ("harness_samples", "harness_test_samples") if cfg.get(k) is not None]
    if clash:
        raise ValueError(f"config['harness_split'] fills the selection and test slices; remove {clash}.")
    split = load_harness_split(str(name))
    tasks = [str(t) for t in (cfg.get("harness_tasks") or [])] if not isinstance(cfg.get("harness_tasks"), str) else [
        t.strip() for t in cfg["harness_tasks"].split(",") if t.strip()]
    if tasks and tasks != [split["task"]]:
        raise ValueError(f"harness_split {name!r} is a split of {split['task']!r}; harness_tasks is {tasks}.")
    record = {"name": str(name), "sha256": split["sha256"], "task": split["task"],
              "n_selection": len(split["selection"]), "n_test": len(split["test"]),
              "n_calibration": len(split["calibration"]), "seed": split["seed"]}
    return {split["task"]: list(split["selection"])}, {split["task"]: list(split["test"])}, record


def check_calibration_disjoint(split_samples: Mapping[str, Sequence[int]], calibration_eval_samples: Mapping[str, Sequence[int]]) -> None:
    """The run's calibration documents (all documents not in its own hold-out) must not be selection/test documents:
    selection + test must lie inside the run's hold-out ``calibration_eval_samples``."""
    for task, used in split_samples.items():
        holdout = set(calibration_eval_samples.get(task, ()))
        leaked = sorted(set(used) - holdout)
        if leaked:
            raise ValueError(
                f"harness_split: {len(leaked)} {task} documents of the selection/test slices are calibration "
                f"documents of this run (e.g. {leaked[:5]}); reduce the calibration pool "
                "(calibration_n_sequences: 40 for ifeval_k_v1)."
            )
