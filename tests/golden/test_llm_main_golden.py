"""Characterization pins for ``eval/llm_rebase`` ``main()`` (Phase 7 / S10a safety net).

Drives the *real* ``main()`` end to end on a tiny offline world (real ``Qwen2ForCausalLM`` source and target built
from config, whitespace tokenizer built in memory) and pins what it produces:

* the run summary (``run_logger.log_summary``) via ``_hashing.hash_json`` (tmp-dir prefix normalised);
* the ``resolved_config`` handed to ``start_run``;
* the ``.pt`` files written under the run dir (``save_merged``);
* the harness call log (tasks, shots, limit, samples, depth and a hash of the *evaluated weights* per call), the
  ``TextLM.build`` log and the tuned-checkpoint load log;
* the error table (exception type, message, whether it fired before any model build / ``start_run``).

Faked, BY NAME across every module in ``PATCH_MODULES`` (``raising=False``, so the same file also runs against the
pre-package commit where ``eval.llm_rebase`` was one module and the harness lived in ``eval.lm_harness_runner``):
``TextLM.build`` (tiny real models), ``load_aligned_tuned_from_ref`` / ``load_ckpt`` (seeded perturbations of the
base state dict), ``build_nli_task_data``, the harness ``run`` (weight-sensitive scorer: sigmoid of the negative
cross-entropy of the evaluated model on fixed token ids) and ``start_run`` (recorder). Calibration text comes from
``config['calibration_prompts']`` (the real resolver on in-memory texts); one case swaps ``resolve_calibration_texts``
to exercise the lm-harness hold-out plumbing. Everything else is real code.

Regenerating (only after establishing that a change is intended; see HASHES.md):
``GOLDEN_CAPTURE=/path/out.txt pytest tests/golden/test_llm_main_golden.py -q -p no:cacheprovider``.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import zlib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from _llm_fixtures import CALIB_TEXTS, local_tokenizer, tiny_qwen2, tiny_qwen3

from ._hashing import deterministic_cpu, hash_json, hash_tensor_dict

PATCH_MODULES: list[str] = [
    "merge_and_rebase.eval.llm_rebase",
    "merge_and_rebase.eval.llm_rebase.cli",
    "merge_and_rebase.eval.llm_rebase.context",
    "merge_and_rebase.eval.llm_rebase.merge",
    "merge_and_rebase.eval.llm_rebase.alpha_search",
    "merge_and_rebase.eval.llm_rebase.summary",
    "merge_and_rebase.eval.llm_rebase.artifacts",
    "merge_and_rebase.eval.llm_rebase.harness",
    "merge_and_rebase.eval.llm_rebase.common",
    # pre-package homes of the same code (S10a moved them); absent modules are skipped
    "merge_and_rebase.eval.llm_common",
    "merge_and_rebase.eval.lm_harness_runner",
    "merge_and_rebase.eval.llm_eval_only",
]

_NLI_WORDS = "entailment neutral contradiction premise hypothesis label snli rte cats sleep dogs bark the a is"
TOKENIZER_TEXTS = [*CALIB_TEXTS, _NLI_WORDS]


@pytest.fixture(autouse=True)
def _deterministic():
    with deterministic_cpu(0):
        yield


# --------------------------------------------------------------------------------------
# The fake world
# --------------------------------------------------------------------------------------


class World:
    """Source/target geometry plus knobs the tests perturb."""

    def __init__(
        self,
        *,
        src_layers: int = 2,
        tgt_layers: int = 2,
        src_kv: int = 2,
        tgt_kv: int = 2,
        tgt_hidden: int = 32,
        tgt_heads: int = 4,
        tuned_scale: float = 0.05,
        src_family: str = "qwen2",
        tgt_family: str = "qwen2",
    ) -> None:
        self.src = dict(layers=src_layers, kv_heads=src_kv, seed=0, family=src_family)
        self.tgt = dict(
            layers=tgt_layers, kv_heads=tgt_kv, hidden=tgt_hidden, heads=tgt_heads, seed=1, family=tgt_family
        )
        self.tuned_scale = tuned_scale


class _Recorder:
    def __init__(self, metadata: dict[str, Any]):
        self.metadata = deepcopy(metadata)
        self.summary: dict[str, Any] | None = None
        self.status: str | None = None
        self.error: Any = None

    def log_event(self, *a, **k):  # not used by llm_rebase; kept for interface parity
        pass

    def log_summary(self, summary_dict):
        self.summary = deepcopy(summary_dict)

    def finish(self, status, error=None):
        self.status = status
        self.error = error


class Calls:
    def __init__(self) -> None:
        self.recorders: list[_Recorder] = []
        self.builds: list[dict[str, Any]] = []
        self.tuned: list[dict[str, Any]] = []
        self.base_ckpts: list[str] = []
        self.harness: list[dict[str, Any]] = []


def _seeded_perturb(sd: dict[str, torch.Tensor], seed: int, scale: float) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for k in sorted(sd):
        v = sd[k]
        out[k] = v + scale * torch.randn(v.shape, generator=gen, dtype=v.dtype) if v.is_floating_point() else v.clone()
    if (
        "lm_head.weight" in out
        and "model.embed_tokens.weight" in out
        and torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
    ):
        out["lm_head.weight"] = out["model.embed_tokens.weight"].clone()
    return out


def _eval_ids(label: str, vocab: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(zlib.crc32(label.encode()))
    return torch.randint(2, vocab, (3, 10), generator=gen)


def _install_fakes(monkeypatch, world: World, calls: Calls, *, fake_calibration: bool = False) -> None:
    from merge_and_rebase.data.llm_calibration import CalibrationTexts
    from merge_and_rebase.data.text_loaders import NLIExample, NLITaskData
    from merge_and_rebase.models.text_lm import TextLM

    def build(cfg):
        role = "src" if "src" in str(cfg.model_name_or_path) else "tgt"
        calls.builds.append(
            {"name": str(cfg.model_name_or_path), "arch": cfg.model_arch, "device": cfg.device, "dtype": cfg.dtype}
        )
        spec = world.src if role == "src" else world.tgt
        builder = tiny_qwen3 if spec.get("family") == "qwen3" else tiny_qwen2
        model = builder(
            layers=spec["layers"],
            kv_heads=spec["kv_heads"],
            hidden=spec.get("hidden", 32),
            heads=spec.get("heads", 4),
            seed=spec["seed"],
        )
        return TextLM(model, local_tokenizer(TOKENIZER_TEXTS))

    def load_aligned_tuned_from_ref(*, ckpt_ref, base_sd, build_cfg, model, prefer_lora_view=False, **_k):
        calls.tuned.append({"ref": str(ckpt_ref), "prefer_lora_view": prefer_lora_view})
        seed = 100 + zlib.crc32(str(ckpt_ref).encode()) % 1000
        return _seeded_perturb(dict(base_sd), seed, world.tuned_scale)

    def load_ckpt(path, *a, **k):
        calls.base_ckpts.append(str(path))
        model = tiny_qwen2(seed=0 if "src" in str(path) else 1)
        return _seeded_perturb({k_: v.detach().clone() for k_, v in model.state_dict().items()}, 7, 0.01)

    def harness_run(tasks, model, tokenizer, device="cuda", num_fewshot=0, batch_size="auto", limit=None, samples=None):
        calls.harness.append(
            {
                "tasks": list(tasks),
                "num_fewshot": num_fewshot,
                "batch_size": batch_size,
                "limit": limit,
                "samples": samples,
                "depth": int(model.config.num_hidden_layers),
                "weights": hash_tensor_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            }
        )
        out: dict[str, float] = {}
        with torch.no_grad():
            for t in tasks:
                ids = _eval_ids(t, int(model.config.vocab_size))
                logits = model(input_ids=ids).logits[:, :-1].float()
                ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
                out[f"{t}_acc"] = float(torch.sigmoid(-(ce - 4.0)))
        return out

    def build_nli_task_data(*, task, split, max_samples=None):
        ex = [NLIExample(f"cats sleep {i}", "dogs bark" if i % 2 else "the a is", i % 3) for i in range(6)]
        return NLITaskData(
            task=task,
            examples=ex,
            labels=["entailment", "neutral", "contradiction"],
            label_texts=["entailment", "neutral", "contradiction"],
            meta={"split": split, "n": len(ex)},
        )

    def start_run(*, entrypoint, logging_cfg, metadata, summary_path=None):
        rec = _Recorder(metadata)
        calls.recorders.append(rec)
        return rec

    def resolve_calibration_texts(*, prompts=None, **_kw):
        return CalibrationTexts(
            list(prompts or CALIB_TEXTS),
            source="fake_harness_corpus",
            eval_samples={"arc_easy": [0, 1, 2], "piqa": [3, 4]},
        )

    monkeypatch.setattr(TextLM, "build", staticmethod(build))
    fakes: dict[str, Any] = {
        "load_aligned_tuned_from_ref": load_aligned_tuned_from_ref,
        "load_ckpt": load_ckpt,
        "build_nli_task_data": build_nli_task_data,
        "start_run": start_run,
        "run": harness_run,
    }
    if fake_calibration:
        fakes["resolve_calibration_texts"] = resolve_calibration_texts
    for module_name in PATCH_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for name, fake in fakes.items():
            # ``run`` is only a harness entry point on the harness modules (a bare ``run`` elsewhere is harmless but
            # pointless); the package ``__init__`` re-exports cli's namespace, so patch it there too by name.
            if name == "run" and not module_name.endswith(("harness", "lm_harness_runner")):
                continue
            monkeypatch.setattr(module, name, fake, raising=False)


# --------------------------------------------------------------------------------------
# Running main()
# --------------------------------------------------------------------------------------


def _norm(obj: Any, root: Path) -> Any:
    return json.loads(json.dumps(obj, default=repr).replace(str(root), "<ROOT>"))


def _base_cfg(root: Path, **overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "source_model_name_or_path": "fake_src",
        "target_model_name_or_path": "fake_tgt",
        "device": "cpu",
        "seed": 3,
        "method": "theseus",
        "tuned_bodies": ["tuned_A", "tuned_B"],
        "harness_tasks": "arc_easy,piqa",
        "calibration_prompts": list(CALIB_TEXTS),
        "calibration_batch_size": 2,
        "calibration_max_length": 12,
        "method_params": {"num_batches": 2, "seq_align": "mean", "verbose": False, "show_progress": False},
        "block_extension_params": {"n_batches_act": 2, "verbose": False, "show_progress": False},
        "alpha": 1.0,
        "logging": {"local_log_dir": str(root / "logs")},
    }
    cfg.update(overrides)
    return cfg


def _launch(cfg: dict[str, Any], root: Path, monkeypatch, world: World, *, fake_calibration: bool = False) -> Calls:
    from merge_and_rebase.eval import llm_rebase

    calls = Calls()
    _install_fakes(monkeypatch, world, calls, fake_calibration=fake_calibration)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["llm_rebase", "--config", str(config_path)])
    try:
        llm_rebase.main()
    except BaseException:
        _launch.last_calls = calls  # type: ignore[attr-defined]
        raise
    return calls


def _tree_hash(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.pt")):
        out[path.relative_to(root).as_posix()] = hash_tensor_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    return out


def run_case(cfg: dict[str, Any], root: Path, monkeypatch, world: World, **kw: Any) -> dict[str, str]:
    calls = _launch(cfg, root, monkeypatch, world, **kw)
    assert len(calls.recorders) == 1
    rec = calls.recorders[0]
    assert rec.status == "success" and rec.summary is not None
    digests = {
        "summary": hash_json(_norm(rec.summary, root)),
        "resolved_config": hash_json(_norm(rec.metadata["resolved_config"], root)),
        "harness_calls": hash_json(calls.harness),
        "builds": hash_json(calls.builds),
        "tuned_loads": hash_json({"tuned": calls.tuned, "base": calls.base_ckpts}),
    }
    for rel, digest in _tree_hash(root).items():
        digests[f"file:{rel}"] = digest
    run_case.last_summary = rec.summary  # type: ignore[attr-defined]
    return digests


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------

_SEARCH = {"alpha_search": True, "alpha_min": 0.0, "alpha_max": 1.0, "alpha_step": 0.5}
_BE = {"n_batches_act": 2, "verbose": False, "show_progress": False}
_SAME = {}
_EXT = {"tgt_layers": 3}
_SHR = {"src_layers": 3}
_GQA = {"tgt_kv": 1}


def _merged(root: Path) -> dict[str, Any]:
    return {"save_merged": str(root / "merged" / "m.pt")}


CASES: dict[str, tuple[dict[str, Any], Callable[[Path], dict[str, Any]]]] = {
    "theseus_same_depth_fixed_alpha": (_SAME, lambda r: {}),
    "theseus_same_depth_alpha_search_save_merged_base_ckpts": (
        _SAME,
        lambda r: {**_SEARCH, **_merged(r), "source_base_ckpt": "ckpt_src", "target_base_ckpt": "ckpt_tgt"},
    ),
    "theseus_same_depth_weights_and_limit": (
        _SAME,
        lambda r: {"weights": [0.7, 0.3], "harness_limit": 5, "harness_num_fewshot": 2, "harness_batch_size": "4"},
    ),
    "theseus_extend_brace_skip_correction_true": (
        _EXT,
        lambda r: {"block_extension_params": {**_BE, "skip_correction": True}},
    ),
    "theseus_extend_brace_skip_correction_false": (
        _EXT,
        lambda r: {"block_extension_params": {**_BE, "skip_correction": False}},
    ),
    "theseus_extend_defaults_alpha_search": (_EXT, lambda r: _SEARCH),
    "theseus_extend_eval_before_rebase": (
        _EXT,
        lambda r: {"eval_before_rebase": True, "eval_source_before_extension": True},
    ),
    "theseus_same_depth_eval_before_rebase": (_SAME, lambda r: {"eval_before_rebase": True}),
    "theseus_shrink_per_weight": (
        _SHR,
        lambda r: {"block_extension_params": {**_BE, "extension_strategy": "interpolate_per_weight"}},
    ),
    "theseus_gqa_same_depth": (_GQA, lambda r: {"method": "theseus_gqa"}),
    "theseus_gqa_extend": (
        {**_GQA, **_EXT},
        lambda r: {"method": "theseus_gqa", "block_extension_params": {**_BE, "skip_correction": True}},
    ),
    "bico_same_depth": (_SAME, lambda r: {"method": "bico", "method_params": {"num_batches": 2, "seq_align": "mean"}}),
    "bico_extend": (_EXT, lambda r: {"method": "bico", "method_params": {"num_batches": 2, "seq_align": "mean"}}),
    "dnm_uncorrected_extend": (
        _EXT,
        lambda r: {"delta_norm_match": "uncorrected", "block_extension_params": {**_BE, "skip_correction": False}},
    ),
    "dnm_uncorrected_extend_uncorrected_source": (
        _EXT,
        lambda r: {
            "delta_norm_match": "uncorrected",
            "transport_delta_source": "uncorrected",
            "block_extension_params": {**_BE, "skip_correction": False},
        },
    ),
    "dnm_literal_none_extend": (
        _EXT,
        lambda r: {"delta_norm_match": "none", "block_extension_params": {**_BE, "skip_correction": False}},
    ),
    "dnm_uncorrected_same_depth": (_SAME, lambda r: {"delta_norm_match": "uncorrected"}),
    "alpha_search_sobol": (
        _SAME,
        lambda r: {
            "alpha_search": True,
            "hyperparam_search": {
                "strategy": "sobol",
                "num_samples": 2,
                "refinement_steps": 1,
                "alpha": {"min": 0.0, "max": 1.0, "step": 0.5},
            },
        },
    ),
    "alpha_search_discrete_values": (
        _SAME,
        lambda r: {"hyperparam_search": {"alpha": {"values": [0.25, 0.75]}}},
    ),
    "save_merged_extend": (_EXT, lambda r: {**_merged(r), "block_extension_params": {**_BE, "skip_correction": True}}),
    "eval_before_rebase_only_extend": (_EXT, lambda r: {"eval_before_rebase_only": True}),
    "eval_before_rebase_only_same_depth": (_SAME, lambda r: {"eval_before_rebase_only": True}),
    "nli_prompt_search_save_merged": (
        _SAME,
        lambda r: {
            "harness_tasks": None,
            "suite": "nli6",
            "tasks": "snli,rte",
            "tuned_bodies": {"snli": "tuned_A", "rte": "tuned_B"},
            "allow_prompt_eval": True,
            "eval_mode": "prompt",
            "fine_tuned_acc": {"snli": 0.5, "rte": 0.6},
            **_SEARCH,
            **_merged(r),
        },
    ),
}

# lm-harness hold-out plumbing (resolver faked: returns eval_samples), harness_test_samples re-scoring
CASES["theseus_holdout_samples_and_test_slice"] = (
    _SAME,
    lambda r: {"harness_test_samples": {"arc_easy": [10, 11], "piqa": [12]}, **_SEARCH},
)
FAKE_CALIBRATION = {"theseus_holdout_samples_and_test_slice"}

# P7.S10b (declared): depth-mismatched THESEUS/BiCo cases pin the legacy depth rule explicitly; the per-method
# defaults, Ariadne (both spellings) and the BiCo discrete index match get their own cases.
_LEGACY_DEPTH_CASES = (
    "bico_extend",
    "eval_before_rebase_only_extend",
    "theseus_extend_defaults_alpha_search",
    "theseus_extend_eval_before_rebase",
    "theseus_shrink_per_weight",
)
for _name in _LEGACY_DEPTH_CASES:
    _sizes, _make = CASES[_name]
    CASES[_name] = (_sizes, lambda r, _make=_make: {**_make(r), "depth_defaults": "legacy"})
_ARIADNE = {"preset": "ariadne", "num_batches": 2, "seed": 0}
CASES["theseus_extend_depth_defaults_method"] = (_EXT, lambda r: {"depth_defaults": "method"})
CASES["bico_extend_discrete_index_match"] = (
    _EXT,
    lambda r: {"method": "bico", "method_params": {"num_batches": 2, "seq_align": "mean"}, "depth_defaults": "method"},
)
CASES["ariadne_same_depth"] = (_SAME, lambda r: {"method": "ariadne", "ariadne_params": dict(_ARIADNE)})
CASES["ariadne_extend"] = (_EXT, lambda r: {"method": "ariadne", "ariadne_params": dict(_ARIADNE)})
CASES["direct_residual_spelling_same_depth"] = (
    _SAME,
    lambda r: {"method": "direct_residual", "direct_residual_params": dict(_ARIADNE)},
)

EXPECTED: dict[str, str] = {
    "alpha_search_discrete_values:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "alpha_search_discrete_values:harness_calls": "64938befbf4cc9e0806ae1b095f264994a81216a55218dd8a622374d35ff48c2",
    "alpha_search_discrete_values:resolved_config": "36de8e57bd3712d633ee93affb8b75c6dfb3e5899bc794c6462069b252d1087a",
    "alpha_search_discrete_values:summary": "5afe59302be3f6c18ad94231dd6a3779f38556ceaa221c48d6595652fd6e8bb3",
    "alpha_search_discrete_values:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "alpha_search_sobol:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "alpha_search_sobol:harness_calls": "9221dd2e9ba9cb8883d58da42eefff09688654ad02185f7f67f9eb8d0a114d1b",
    "alpha_search_sobol:resolved_config": "d122c52c84d81f38cd0a007806ff8cbb9845537dd644456e9d5db981fb1ce907",
    "alpha_search_sobol:summary": "860a96e6a03b84bbbe68536d9345b41bc112ece16af8661198a961bfdc802cfe",
    "alpha_search_sobol:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "ariadne_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "ariadne_extend:harness_calls": "cf893107056aceda30eda6c7e3fd20c71af2806ca74528cd821341a0095f327c",
    "ariadne_extend:resolved_config": "2a21726f00184f9ff4966c248ddb7d53ecf5c5727dc9b03f71720cbfd11149d1",
    "ariadne_extend:summary": "3f57d8525bd2ee7221c098123b66212b24c9b25add7a1e7176e5733e9f6f6b29",
    "ariadne_extend:tuned_loads": "8823381edc53b384c1713eb184d37b087cb1c45c8edbde9b015d07d18f65f6ff",
    "ariadne_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "ariadne_same_depth:harness_calls": "4c6450b4b83b1f327d8187f0a9f146ac88f29bfa6746cc43d2d42e07f7915c7c",
    "ariadne_same_depth:resolved_config": "2a21726f00184f9ff4966c248ddb7d53ecf5c5727dc9b03f71720cbfd11149d1",
    "ariadne_same_depth:summary": "ff819d32420df17f70620244e34c2f64e329b28f950449c89695c3478b5eb0c6",
    "ariadne_same_depth:tuned_loads": "8823381edc53b384c1713eb184d37b087cb1c45c8edbde9b015d07d18f65f6ff",
    "bico_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "bico_extend:harness_calls": "64544190ae69c1319bd790596b9a055a3c164d25d4505e2dce7236118d02eff4",
    "bico_extend:resolved_config": "17cb46ac0f81d1f4b78ed8f1014ec5ff6684c9f0bdf4f92e237167144ebf16c9",
    "bico_extend:summary": "66b43c9915b6cbe9d433d543b8be9aa98b1e62640bd2aa2c35a9553b65435a48",
    "bico_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "bico_extend_discrete_index_match:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "bico_extend_discrete_index_match:harness_calls": "5c518c1dc7d10512c0e204e1fd0bacd354dc3bd978d1930d8e1c6881cfcc9a9d",
    "bico_extend_discrete_index_match:resolved_config": "7ee8f638bb5a27aa8cfcad1a41347553cd340b100e3790fa5b9499ad80745df6",
    "bico_extend_discrete_index_match:summary": "88e4e51f82c0f2f2f10d624ab4330c524e404cd03d5d48ab0d5c6bfe0da97de2",
    "bico_extend_discrete_index_match:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "bico_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "bico_same_depth:harness_calls": "5d6977c69ef9feaa598fc97f77954f3378d4e4953ff8fc209080c6b29e1490ab",
    "bico_same_depth:resolved_config": "cc7f4890139bffff964ccf6261ff7625974cdb9167ee89189823647d123ca87b",
    "bico_same_depth:summary": "d4578f90e32b0092d862511ace8c92029514e0c9074f6881f589360ddd58a78a",
    "bico_same_depth:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "direct_residual_spelling_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "direct_residual_spelling_same_depth:harness_calls": "4c6450b4b83b1f327d8187f0a9f146ac88f29bfa6746cc43d2d42e07f7915c7c",
    "direct_residual_spelling_same_depth:resolved_config": "631bee4c41a6079de1b55dbe23236dad6a76c2dd322fa5c104f41b04db628332",
    "direct_residual_spelling_same_depth:summary": "948c5c81c2dcc9ef6612a670b41dfb06a923158eb964c1ab0cffc26641bbe98c",
    "direct_residual_spelling_same_depth:tuned_loads": "8823381edc53b384c1713eb184d37b087cb1c45c8edbde9b015d07d18f65f6ff",
    "dnm_literal_none_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "dnm_literal_none_extend:harness_calls": "2edcff9eba89b715eae578674b8dfa071a1dd8e6f335ee773020065f86ef84de",
    "dnm_literal_none_extend:resolved_config": "234bc8e188f3aff22430d791dcea123227ae31bd4d96dccca3995a2afbb583de",
    "dnm_literal_none_extend:summary": "a15bcee1bdf68c8b3b4999b2fa983e49a7b709f6d76f92efb1dfba025ae5f7ff",
    "dnm_literal_none_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "dnm_uncorrected_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "dnm_uncorrected_extend:harness_calls": "1422983e1c92010dfc65a9b5f1c45563b1605f6e1e0fc4e672582f37a4055dcd",
    "dnm_uncorrected_extend:resolved_config": "8ddc757da5411dae4be6cfb62e521f6456f176f42f24bcde5663b113c1b03d07",
    "dnm_uncorrected_extend:summary": "440738251ee99999c3765d20a5978c88c01507dffbf468d0407bba804e3f6bb1",
    "dnm_uncorrected_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "dnm_uncorrected_extend_uncorrected_source:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "dnm_uncorrected_extend_uncorrected_source:harness_calls": "b15355e163ae1f0418f68cf326532c231c71c1c603b8872c181cd0a136bec160",
    "dnm_uncorrected_extend_uncorrected_source:resolved_config": "15acd93de5975af81d4d6a2b8a916ad27852cc98629a8be95b42c684d8850ae5",
    "dnm_uncorrected_extend_uncorrected_source:summary": "7ecf2ada71115bb7a8a2aec8904b88d7d5787db49e6e99990f599695539af2cf",
    "dnm_uncorrected_extend_uncorrected_source:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "dnm_uncorrected_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "dnm_uncorrected_same_depth:harness_calls": "432bc79a549313161a46013716953c2ecd83db25693bfd1162ae5f7867392c48",
    "dnm_uncorrected_same_depth:resolved_config": "5ec0968dc769721a9f650dba0398106549fd539ee548556cf2cf2bd4e8cb67ce",
    "dnm_uncorrected_same_depth:summary": "1e26774b50364fea086298416f76ff88f33ff0893a2c0bc90160375d6d884929",
    "dnm_uncorrected_same_depth:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "eval_before_rebase_only_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "eval_before_rebase_only_extend:harness_calls": "223046c5d23c59639535adf8f0a5e7cb32c483849943f68b7f322aefd6954bd7",
    "eval_before_rebase_only_extend:resolved_config": "e871a5a2eb8ea494e7b13259fc7090a933687bcb32b20280cabeb7f3dd59abdb",
    "eval_before_rebase_only_extend:summary": "c159ffabb3087998b817e27a09a783689c037f963c7990d8aeeed50896a139b8",
    "eval_before_rebase_only_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "eval_before_rebase_only_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "eval_before_rebase_only_same_depth:harness_calls": "f5d24f2654f8a49dd2b266b6f31525983004bd4f074edd12231201cd3691d74c",
    "eval_before_rebase_only_same_depth:resolved_config": "2c913ac063c3e59a09537a42a5563bba2052b250f275414f6b7c33eba897d216",
    "eval_before_rebase_only_same_depth:summary": "10118bec28c5af73069fadd45b7d822be6320356fdf2549146a27567a83929fc",
    "eval_before_rebase_only_same_depth:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "nli_prompt_search_save_merged:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "nli_prompt_search_save_merged:file:merged/m.pt": "2d93015915bda0806c8fd6fa36d27af6d844abcae4ff64b3cd7ceb7d1f9e7725",
    "nli_prompt_search_save_merged:harness_calls": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "nli_prompt_search_save_merged:resolved_config": "d9244b4941f2a074682f669dc9db128b355381269a5ebd6c4f64f3bc84f7cfd0",
    "nli_prompt_search_save_merged:summary": "c285adf9ca8837f5957915d57398f1050d422638b1c975f11c7ccea7e4b3bc10",
    "nli_prompt_search_save_merged:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "save_merged_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "save_merged_extend:file:merged/m.pt": "ab21e41da3245e88dc947002ba655bc728555ae80caaa7238138d102f103cb18",
    "save_merged_extend:harness_calls": "f0b1d86b3c98bd8a109aaa657ec3105ec2db683153603f189cab5b5a3303a9a0",
    "save_merged_extend:resolved_config": "9e5f67e25e01bfe5fc15a559655c3972cc07602f8d9a976bca190f65b2c9692c",
    "save_merged_extend:summary": "555dad429a99c13c39f812c530a65628b78e9e6f5c8c10fdbe2ba56a41040954",
    "save_merged_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_extend_brace_skip_correction_false:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_extend_brace_skip_correction_false:harness_calls": "2edcff9eba89b715eae578674b8dfa071a1dd8e6f335ee773020065f86ef84de",
    "theseus_extend_brace_skip_correction_false:resolved_config": "040f876561e3849809d24c9f780e1a7d4d5f869f7a5af128d0d8b0fecd279850",
    "theseus_extend_brace_skip_correction_false:summary": "a15bcee1bdf68c8b3b4999b2fa983e49a7b709f6d76f92efb1dfba025ae5f7ff",
    "theseus_extend_brace_skip_correction_false:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_extend_brace_skip_correction_true:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_extend_brace_skip_correction_true:harness_calls": "f0b1d86b3c98bd8a109aaa657ec3105ec2db683153603f189cab5b5a3303a9a0",
    "theseus_extend_brace_skip_correction_true:resolved_config": "9dc1038bed495b51e06afd11b8795aee702381d12518065e58aa59293c2ad2c0",
    "theseus_extend_brace_skip_correction_true:summary": "555dad429a99c13c39f812c530a65628b78e9e6f5c8c10fdbe2ba56a41040954",
    "theseus_extend_brace_skip_correction_true:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_extend_defaults_alpha_search:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_extend_defaults_alpha_search:harness_calls": "8eedb83aa039f9c765b7ea9ce55f53b41b7f6e9a4b50b9939eca8cca0a933b1b",
    "theseus_extend_defaults_alpha_search:resolved_config": "89c8491afd1bbcc81a70c8c0ec424bec7aa6d43f09eea7904dbfa14f1d93bcb6",
    "theseus_extend_defaults_alpha_search:summary": "ae4b65827bdadf515eac50b7448fbaeb5ce89838dcc5ed6e35a3bb797493a698",
    "theseus_extend_defaults_alpha_search:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_extend_depth_defaults_method:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_extend_depth_defaults_method:harness_calls": "f0b1d86b3c98bd8a109aaa657ec3105ec2db683153603f189cab5b5a3303a9a0",
    "theseus_extend_depth_defaults_method:resolved_config": "b963297cb59365f7a44a32ffd3bcd07fc5921911a47e0993155fe01b0c54be00",
    "theseus_extend_depth_defaults_method:summary": "555dad429a99c13c39f812c530a65628b78e9e6f5c8c10fdbe2ba56a41040954",
    "theseus_extend_depth_defaults_method:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_extend_eval_before_rebase:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_extend_eval_before_rebase:harness_calls": "ba07e46f2425169faf6d2e1e8088c7f8cd4f11e60829dfd974ff29869a07f23c",
    "theseus_extend_eval_before_rebase:resolved_config": "1820cb76f65eb850ffa8aa5d7b95445a87347a57d6077e82c0a78137ba8c424c",
    "theseus_extend_eval_before_rebase:summary": "f2bd3158355549ea3659d025156a1d0d73c739cf987bb4fff5820ff8a81d4e95",
    "theseus_extend_eval_before_rebase:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_gqa_extend:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_gqa_extend:harness_calls": "84a8f7f877d325c5717538ca69be0fb4eb0e18f9151a0876272b113021422e62",
    "theseus_gqa_extend:resolved_config": "86334668f847c660e6c0e4d0e4efeb46c7bf3c884af6182e9b8036924860e4d3",
    "theseus_gqa_extend:summary": "bf9a16a777d83e5a2b9f4da382dacbc8edd739020529b4e7164b80a38b6d17ed",
    "theseus_gqa_extend:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_gqa_same_depth:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_gqa_same_depth:harness_calls": "605b13331aebb9175942c10a04a5795693fa2e24a1fe1994d0afd25e928725f0",
    "theseus_gqa_same_depth:resolved_config": "040c95e1251a4960e3707595c8eb3a6a436b654031e092478d1e11e8e03dd72b",
    "theseus_gqa_same_depth:summary": "e431282c374918d95d68c1dcd173e9f26dfe202b99d0d030b8793d8ff5925fab",
    "theseus_gqa_same_depth:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_holdout_samples_and_test_slice:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_holdout_samples_and_test_slice:harness_calls": "e202a4d8e7daade92d194e555600f2971287a8447bcf5ee4f6c7087b828a3bcb",
    "theseus_holdout_samples_and_test_slice:resolved_config": "25a5c8779c7620667e0261f331ee9346d696524f976d5aee910bb2309b0a7c32",
    "theseus_holdout_samples_and_test_slice:summary": "f96e033e339ef9bb1a2cf6ec9238dbcca4b6cf485e98dc7b418e308db20bdc43",
    "theseus_holdout_samples_and_test_slice:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:file:merged/m.pt": "82de848fc24bba193a42a6b44257ebd3b3b0a743f222fc1cca8de5f40c713939",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:harness_calls": "73641f46da9d5c62e2bc3c7745a8442c4cb66b5ac1f1c8b25275d1b8601010aa",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:resolved_config": "7fdd0c581142e45d045d0aeeeb289736ccf2a3f3e311751f88d0a629fbb14736",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:summary": "0043fd7c27bc7fb55f9b90697af2f6a06b7a852ae773a81d479ae84c6439cdc5",
    "theseus_same_depth_alpha_search_save_merged_base_ckpts:tuned_loads": "e9ff6a9cd8a5142be72f7a033378f13e9e79c8c28f2471eb2dacdb4922f8754a",
    "theseus_same_depth_eval_before_rebase:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_same_depth_eval_before_rebase:harness_calls": "e78b7704903671a8113a9e326390f36e9b3b6fe54ee66784af1d0a2bda2605e9",
    "theseus_same_depth_eval_before_rebase:resolved_config": "fe10b709bb2bb7fa9a86159816dee2bc4e303c595cfb02ba08e0509839307bce",
    "theseus_same_depth_eval_before_rebase:summary": "93e689aa7eadaae7b33f00a3adb7465463ddc6b8bfcb0a8f122b5847636dec1c",
    "theseus_same_depth_eval_before_rebase:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_same_depth_fixed_alpha:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_same_depth_fixed_alpha:harness_calls": "b089508a52380b79a41f35c6f8033cf7310fa7a2ea17446a3c219c31a2576336",
    "theseus_same_depth_fixed_alpha:resolved_config": "3b648e82ad1d223ded45ae488d93e060133927886e860c9c6a85c81f0c1c1065",
    "theseus_same_depth_fixed_alpha:summary": "771f0da1d74f570e452c1e3d06aca266019a0483a57ac1637b964dae3b24a982",
    "theseus_same_depth_fixed_alpha:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_same_depth_weights_and_limit:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_same_depth_weights_and_limit:harness_calls": "221031eeeb72e358ac6f64638090b0845c15ee54c1b15ae5fb88e4c8bfa0f4e4",
    "theseus_same_depth_weights_and_limit:resolved_config": "363852441c54bc27ca907cd50adf78dd0e971aee39bb8b847f0dba54dc56d49f",
    "theseus_same_depth_weights_and_limit:summary": "c1104c73b57bca3a725fc7a73e4729aa7d1c4dc8733af839a457a53888d88934",
    "theseus_same_depth_weights_and_limit:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
    "theseus_shrink_per_weight:builds": "40b826545ffac45f8c89117af922984a7e5ae7b5943c5e9d0e833b5e1027bd56",
    "theseus_shrink_per_weight:harness_calls": "ebc92af2e9b8edf1e82fd77e0f4d0329139cf162d2de48ef6bf9fbe16b783542",
    "theseus_shrink_per_weight:resolved_config": "a9884dc5a30c103f478dca3063e11d33d63a94be56439b73c295661a3a97eb36",
    "theseus_shrink_per_weight:summary": "089790e2c77f6222aac5da19cce6a984c7fa289d0dfa78f92341c4d52f678ca5",
    "theseus_shrink_per_weight:tuned_loads": "a1ee106db8d03412d6b1bebfdc71b2419905a4ce536a6254ebe11f7d44d20319",
}
EXPECTED_ERRORS: dict[str, list[Any]] = {
    "theseus_extend_depth_defaults_guard": [
        "ConfigMeaningChangedError",
        'theseus: depth-mismatched pair without an explicit skip_correction. The per-method depth defaults would change this run\'s result. Either keep the previous behaviour with "block_extension_params": {"skip_correction": false} (or "depth_defaults": "legacy"), or accept the new method default with "depth_defaults": "method".',
        2,
        1,
        ["failed"],
    ],
    "alpha_step_nonpositive": ["ValueError", "alpha_step must be > 0.", 2, 1, ["failed"]],
    "block_extension_bad_strategy": [
        "ValueError",
        "Unsupported extension_strategy 'bogus'. Expected: interpolate, per_weight, shrink, interpolate_per_weight, duplicate_per_weight.",
        2,
        1,
        ["failed"],
    ],
    "calibration_n_sequences_zero": [
        "ValueError",
        "config['calibration_n_sequences'] must be > 0 when given.",
        2,
        1,
        ["failed"],
    ],
    "calibration_prompts_not_list": [
        "ValueError",
        "config['calibration_prompts'] must be a list of strings, or a path to a JSON file holding one.",
        2,
        1,
        ["failed"],
    ],
    "delta_norm_match_bad": [
        "ValueError",
        "delta_norm_match must be null or 'uncorrected'. Got: 'foo'",
        2,
        1,
        ["failed"],
    ],
    "depth_mismatch_block_extension_disabled": [
        "ValueError",
        "Same-size models have different depths (2 vs 3). Depth mismatch requires block-extension prealign.",
        2,
        1,
        ["failed"],
    ],
    "eval_before_rebase_only_no_harness": [
        "ValueError",
        "eval_before_rebase_only needs harness_tasks: there is nothing else to run.",
        2,
        1,
        ["failed"],
    ],
    "harness_samples_bad_indices": [
        "ValueError",
        "config['harness_samples'] values must be lists of non-negative document indices.",
        2,
        1,
        ["failed"],
    ],
    "harness_samples_disagree_with_holdout": [
        "ValueError",
        "config['harness_samples'] disagrees with the IFEval hold-out derived from calibration; use the derived samples or an independent calibration corpus.",
        2,
        1,
        ["failed"],
    ],
    "harness_samples_not_dict": [
        "ValueError",
        "config['harness_samples'] must map task names to document-index lists.",
        2,
        1,
        ["failed"],
    ],
    "harness_test_samples_not_dict": [
        "ValueError",
        "config['harness_test_samples'] must map task names to index lists.",
        2,
        1,
        ["failed"],
    ],
    "harness_test_samples_overlap": [
        "ValueError",
        "harness_test_samples overlaps the alpha-search slice for {'arc_easy': 1}; the reported number would be selected on documents it is scored on.",
        2,
        1,
        ["failed"],
    ],
    "method_params_n_batches": [
        "ValueError",
        "config['method_params'].n_batches is deprecated: it silently raced with method_params.num_batches (whichever the resolver checked first won, so the other was ignored without warning). Rename it to 'num_batches' in the config.",
        0,
        0,
        [],
    ],
    "missing_target_model": [
        "ValueError",
        "Both source_model_name_or_path and target_model_name_or_path are required.",
        0,
        0,
        [],
    ],
    "prompt_eval_not_allowed": [
        "ValueError",
        "Prompt evaluation is disabled unless explicitly enabled. Set --allow-prompt-eval (or config['allow_prompt_eval']=true).",
        2,
        1,
        ["failed"],
    ],
    "shrink_non_per_weight": [
        "ValueError",
        "LLM depth shrinking requires a per-weight block-extension strategy. Got 'duplicate'.",
        2,
        1,
        ["failed"],
    ],
    "transport_delta_source_bad": [
        "ValueError",
        "transport_delta_source must be 'corrected' or 'uncorrected'. Got: 'foo'",
        2,
        1,
        ["failed"],
    ],
    "tuned_bodies_missing": [
        "ValueError",
        "tuned_bodies config is required (dict task->path or list).",
        2,
        1,
        ["failed"],
    ],
    "tuned_bodies_missing_task_key": [
        "ValueError",
        "tuned_bodies missing task keys: ['rte']. Provided: ['snli']",
        2,
        1,
        ["failed"],
    ],
    "tuned_bodies_wrong_type": ["ValueError", "tuned_bodies must be a dict or list.", 2, 1, ["failed"]],
    "unknown_method": [
        "KeyError",
        "\"Unknown rebase method 'nope'. Available: ['ariadne', 'bico', 'bico_gradin', 'direct_residual', 'gradfix', 'identity', 'orthogonal_shift', 'theseus', 'theseus_gqa', 'transfusion']\"",
        0,
        0,
        [],
    ],
    "unknown_suite": ["ValueError", "Unknown suite 'foo'. Available: ['nli6']", 2, 1, ["failed"]],
}


def _capture(lines: dict[str, str]) -> bool:
    capture = os.environ.get("GOLDEN_CAPTURE")
    if not capture:
        return False
    with open(capture, "a") as fh:
        for name, digest in lines.items():
            fh.write(f"{name} {digest}\n")
    return True


@pytest.mark.parametrize("name", sorted(CASES))
def test_llm_main_golden(name, tmp_path, monkeypatch):
    world_kw, make_cfg = CASES[name]
    root = tmp_path / "run"

    def once(sub: str) -> dict[str, str]:
        r = tmp_path / sub
        return run_case(
            _base_cfg(r, **make_cfg(r)), r, monkeypatch, World(**world_kw), fake_calibration=name in FAKE_CALIBRATION
        )

    first = once("run")
    second = once(
        "run"
    )  # same dir name: identical resolved paths, a fresh tmp world is not needed (guard is vision-only)
    assert first == second, f"{name}: two in-process runs differ"
    del root
    if _capture({f"{name}:{k}": v for k, v in first.items()}):
        return
    got = {f"{name}:{k}": v for k, v in first.items()}
    expected = {k: v for k, v in EXPECTED.items() if k.startswith(f"{name}:")}
    assert got == expected


# --------------------------------------------------------------------------------------
# Structural assertions that do not depend on hashes
# --------------------------------------------------------------------------------------


def _run(name: str, tmp_path, monkeypatch, **cfg_over):
    world_kw, make_cfg = CASES[name]
    r = tmp_path / "s"
    digests = run_case(
        _base_cfg(r, **{**make_cfg(r), **cfg_over}),
        r,
        monkeypatch,
        World(**world_kw),
        fake_calibration=name in FAKE_CALIBRATION,
    )
    return digests, run_case.last_summary  # type: ignore[attr-defined]


def test_summary_shapes(tmp_path, monkeypatch):
    _, s = _run("theseus_extend_brace_skip_correction_true", tmp_path / "a", monkeypatch)
    assert s["method"] == "theseus" and s["backend"] == "lm_harness" and list(s["harness_results_by_alpha"]) == ["1"]
    assert s["before_rebase_model"] == "extended_source_base" and s["saved_merged_path"] is None
    _, s = _run("eval_before_rebase_only_extend", tmp_path / "b", monkeypatch)
    assert s["stopped_after"] == "before_rebase_eval" and s["before_rebase_model"] == "extended_source_base"
    assert (s["source_depth"], s["target_depth"]) == (2, 3)
    assert set(s["harness_results_before_rebase"]) == {"extended_source_base:task_0", "extended_source_base:task_1"}
    _, s = _run("eval_before_rebase_only_same_depth", tmp_path / "c", monkeypatch)
    assert s["before_rebase_model"] == "source_base" and "arc_easy_acc" in s["harness_results_before_rebase"]


def test_holdout_samples_flow_to_harness(tmp_path, monkeypatch):
    world_kw, make_cfg = CASES["theseus_holdout_samples_and_test_slice"]
    r = tmp_path / "h"
    calls = _launch(_base_cfg(r, **make_cfg(r)), r, monkeypatch, World(**world_kw), fake_calibration=True)
    samples = [c["samples"] for c in calls.harness]
    assert samples[:-1] == [{"arc_easy": [0, 1, 2], "piqa": [3, 4]}] * (len(samples) - 1)
    assert samples[-1] == {"arc_easy": [10, 11], "piqa": [12]}  # test slice rescored once, at the best alpha


def test_alpha_search_visits_every_grid_point(tmp_path, monkeypatch):
    digests, s = _run("theseus_same_depth_alpha_search_save_merged_base_ckpts", tmp_path, monkeypatch)
    assert list(s["harness_results_by_alpha"]) == ["0", "0.5", "1"]
    assert "file:merged/m.pt" in digests and s["saved_merged_path"].endswith("merged/m.pt")


def test_perturbed_tuned_checkpoint_changes_summary(tmp_path, monkeypatch):
    world_kw, make_cfg = CASES["theseus_same_depth_fixed_alpha"]
    out = []
    for scale in (0.05, 0.06):
        r = tmp_path / f"p{scale}"
        out.append(run_case(_base_cfg(r, **make_cfg(r)), r, monkeypatch, World(**world_kw, tuned_scale=scale)))
    assert out[0]["summary"] != out[1]["summary"]
    assert out[0]["harness_calls"] != out[1]["harness_calls"]
    assert out[0]["resolved_config"] == out[1]["resolved_config"]


# --------------------------------------------------------------------------------------
# Error table: [type, message, #model builds, #runs started, recorder statuses]
# --------------------------------------------------------------------------------------

_NLI = {
    "harness_tasks": None,
    "suite": "nli6",
    "tasks": "snli,rte",
    "tuned_bodies": {"snli": "tuned_A", "rte": "tuned_B"},
    "eval_mode": "prompt",
    "allow_prompt_eval": True,
}

ERROR_CASES: dict[str, tuple[dict[str, Any], dict[str, Any], bool]] = {
    # name: (world kwargs, cfg overrides, fake_calibration)
    "missing_target_model": (_SAME, {"target_model_name_or_path": None}, False),
    "unknown_method": (_SAME, {"method": "nope"}, False),
    "method_params_n_batches": (_SAME, {"method_params": {"n_batches": 2}}, False),
    "tuned_bodies_missing": (_SAME, {"tuned_bodies": None}, False),
    "tuned_bodies_wrong_type": (_SAME, {"tuned_bodies": "tuned_A"}, False),
    "tuned_bodies_missing_task_key": (_SAME, {**_NLI, "tuned_bodies": {"snli": "tuned_A"}}, False),
    "prompt_eval_not_allowed": (_SAME, {**_NLI, "allow_prompt_eval": False}, False),
    "unknown_suite": (_SAME, {**_NLI, "suite": "foo"}, False),
    "eval_before_rebase_only_no_harness": (_SAME, {**_NLI, "eval_before_rebase_only": True}, False),
    "calibration_prompts_not_list": (_SAME, {"calibration_prompts": {"a": 1}}, False),
    "calibration_n_sequences_zero": (_SAME, {"calibration_n_sequences": 0}, False),
    "harness_samples_not_dict": (_SAME, {"harness_samples": [1, 2]}, False),
    "harness_samples_bad_indices": (_SAME, {"harness_samples": {"arc_easy": [-1]}}, False),
    "harness_samples_disagree_with_holdout": (_SAME, {"harness_samples": {"arc_easy": [9]}}, True),
    "harness_test_samples_not_dict": (_SAME, {"harness_test_samples": [1]}, False),
    "harness_test_samples_overlap": (_SAME, {"harness_test_samples": {"arc_easy": [1, 20]}}, True),
    "delta_norm_match_bad": (_SAME, {"delta_norm_match": "foo"}, False),
    "transport_delta_source_bad": (_SAME, {"transport_delta_source": "foo"}, False),
    "shrink_non_per_weight": (_SHR, {"block_extension_params": {**_BE, "extension_strategy": "duplicate"}}, False),
    "depth_mismatch_block_extension_disabled": (_EXT, {"block_extension_enabled": False}, False),
    "block_extension_bad_strategy": (
        _EXT,
        {"block_extension_params": {**_BE, "extension_strategy": "bogus"}, "depth_defaults": "legacy"},
        False,
    ),
    "theseus_extend_depth_defaults_guard": (_EXT, {}, False),
    "alpha_step_nonpositive": (_SAME, {"alpha_search": True, "alpha_step": 0.0}, False),
}


@pytest.mark.parametrize("name", sorted(ERROR_CASES))
def test_llm_main_error_table(name, tmp_path, monkeypatch):
    world_kw, over, fake_cal = ERROR_CASES[name]
    r = tmp_path / "e"
    cfg = _base_cfg(r, **over)
    with pytest.raises(Exception) as ei:  # noqa: B017 - the point is to pin type + message
        _launch(cfg, r, monkeypatch, World(**world_kw), fake_calibration=fake_cal)
    calls = _launch.last_calls  # type: ignore[attr-defined]
    row = [
        type(ei.value).__name__,
        str(ei.value).replace(str(r), "<ROOT>"),
        len(calls.builds),
        len(calls.recorders),
        [rec.status for rec in calls.recorders],
    ]
    if os.environ.get("GOLDEN_CAPTURE"):
        with open(os.environ["GOLDEN_CAPTURE"], "a") as fh:
            fh.write(f"ERR {name} {json.dumps(row)}\n")
        return
    assert row == EXPECTED_ERRORS[name]


def test_ariadne_spellings_fit_the_same_vector():
    """`direct_residual` is a pure alias of `ariadne`: identical evaluated weights and saved states."""
    for part in ("harness_calls", "builds", "tuned_loads"):
        a, b = EXPECTED.get(f"ariadne_same_depth:{part}"), EXPECTED.get(f"direct_residual_spelling_same_depth:{part}")
        assert a is not None and a == b, part


@pytest.mark.parametrize(("name", "target_layers"), [("ariadne_same_depth", 2), ("ariadne_extend", 3)])
def test_ariadne_task_vector_is_only_the_down_proj_fit(name, target_layers, tmp_path, monkeypatch):
    """Ariadne's LLM task vector: weight + materialized bias of every target down_proj, nothing else."""
    _, s = _run(name, tmp_path / "a", monkeypatch)
    assert s["merged_delta"]["key_count"] == 2 * target_layers
    tv = s["task_vectors"]
    assert tv["materialized_bias_keys"] == [f"model.layers.{j}.mlp.down_proj.bias" for j in range(target_layers)]
    assert all(row["norm_match_scale"] == 1.0 for row in tv["per_task"])
