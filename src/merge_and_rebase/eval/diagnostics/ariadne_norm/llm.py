"""LLM adapter of the Ariadne norm diagnostics: rebuilds one ``eval.llm_rebase`` Ariadne run's fit inputs.

Everything is built through the same functions the reference run calls (``build_runtime``, ``build_task_models``,
``materialize_missing_projection_biases``, ``build_text_calibration_loader``, ``DiscreteLayerPairing``), in the same
order, so that ``AriadneRebase``'s fit is reproduced on identical models and calibration text.

Config keys (on top of the family-agnostic ones in ``cli``):

- ``reference_summary``: summary JSON of the reference Ariadne run (``llm_ariadne_main_holdout.json``).
- ``reference_config``: the config JSON that run was launched with (the resolved config embedded in the summary is
  checked against it when present).
- ``alpha_star_summary``: summary JSON of the alpha sweep of the same configuration; ``alpha_star`` is its ``best_alpha``.
- ``reference_merged_state``: the run's ``merged_state.pt`` (fp32 full target state, target base + alpha * tau).
- ``tasks``: ``"all"`` or a list of task names (a harness-only run has the single task ``task_0``).
- ``generic_num_batches`` (25), ``generic_dataset`` (``"hellaswag"``), ``generic_seed`` (0): the off-task N6 batches.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.data.llm_calibration import build_text_calibration_loader, resolve_calibration_texts
from merge_and_rebase.models.text_lm import TextBuildConfig
from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import apply_depth_pairing_override
from merge_and_rebase.rebase.methods._ariadne.biases import materialize_missing_projection_biases
from merge_and_rebase.rebase.methods._ariadne.capture import ROW_MASK_KEY, _attach_row_masks
from merge_and_rebase.rebase.methods._ariadne.layouts import _layout_for
from merge_and_rebase.rebase.prestep import StageEnv

from .inputs import DiagnosticInputs

Tensor = torch.Tensor

#: ``generic_dataset`` name -> ``calibration_dataset`` spec of ``resolve_calibration_texts`` (cached offline).
#: Rowan/hellaswag has no column ``_pick_text_column`` recognises; ``ctx`` is the context sentence(s) of a sample.
GENERIC_DATASETS: dict[str, dict[str, str]] = {
    "hellaswag": {"path": "Rowan/hellaswag", "split": "validation", "text_column": "ctx"},
}
#: The generic pool is this many times larger than needed; ``generic_seed`` picks the batches' texts out of it.
_GENERIC_POOL_FACTOR = 10


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bias_key(weight_key: str) -> str:
    return f"{weight_key[: -len('.weight')]}.bias"


def token_index_from_mask(mask: Tensor, n_rows: int) -> Tensor:
    """Within-sequence position (0 = first content token) of each content row, in ``tokens[mask]`` row-major order.

    Position counts content tokens only, so it is right for right- and left-padded batches alike.
    """
    mask = mask.bool().cpu()
    pos = mask.long().cumsum(dim=1) - 1
    out = pos[mask]
    if out.numel() != int(n_rows):
        raise ValueError(f"token_index: mask has {out.numel()} content rows, the capture yields {n_rows}")
    return out


def llm_token_index(batch: Mapping[str, Any], n_rows: int) -> Tensor:
    """``DiagnosticInputs.token_index`` for decoder batches (the combined row mask is attached to every batch)."""
    mask = batch.get(ROW_MASK_KEY) if isinstance(batch, Mapping) else None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("token_index needs a calibration batch carrying the attached row mask")
    return token_index_from_mask(mask, n_rows)


def extract_reference_task_vector(
    merged: Mapping[str, Tensor], target_base_state: Mapping[str, Tensor], keys: Sequence[str]
) -> dict[str, Tensor]:
    """``tau[key] = merged[key] - target_base[key]`` (fp32, CPU) for ``keys`` only.

    ``merged`` is the run's saved target state at ``alpha = 1`` (``target_base + 1.0 * tau`` in fp32), so the
    difference recovers the unit task vector up to fp32 rounding (callers declare ``"tolerance"``).
    """
    out: dict[str, Tensor] = {}
    for key in keys:
        if key not in merged:
            raise KeyError(f"reference merged state has no {key!r}")
        if key not in target_base_state:
            raise KeyError(f"target base state has no {key!r}")
        m, b = merged[key], target_base_state[key]
        if tuple(m.shape) != tuple(b.shape):
            raise ValueError(f"{key}: merged {tuple(m.shape)} vs target base {tuple(b.shape)}")
        out[key] = (m.detach().to("cpu", torch.float32) - b.detach().to("cpu", torch.float32)).clone()
    return out


def _load_reference_task_vector(path: str, target_base_state: Mapping[str, Tensor], keys: Sequence[str]) -> dict:
    merged = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return extract_reference_task_vector(merged, target_base_state, keys)
    finally:
        del merged


def _generic_texts(name: str, n_texts: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    """``n_texts`` generic texts via the repo's own corpus resolver, picked from a larger pool by ``seed``."""
    if name not in GENERIC_DATASETS:
        raise ValueError(f"unknown generic_dataset {name!r}; known: {sorted(GENERIC_DATASETS)}")
    resolved = resolve_calibration_texts(
        calibration_dataset=GENERIC_DATASETS[name], n_sequences=n_texts * _GENERIC_POOL_FACTOR
    )
    pool = resolved.texts
    if len(pool) < n_texts:
        raise ValueError(f"generic pool {name!r} has {len(pool)} texts, need {n_texts}")
    pick = sorted(torch.randperm(len(pool), generator=torch.Generator().manual_seed(int(seed))).tolist()[:n_texts])
    texts = [pool[i] for i in pick]
    return texts, {
        "dataset": name,
        "source": resolved.source,
        "seed": int(seed),
        "pool_size": len(pool),
        "indices_in_pool": pick,
        "texts_sha256": hashlib.sha256("\x00".join(texts).encode()).hexdigest(),
        "n_texts": len(texts),
    }


def _heldout_docs(rt: Any) -> list[dict[str, Any]]:
    """The harness docs the reference run held out of calibration (the ones it scored), rendered as the harness does."""
    from lm_eval.tasks import TaskManager

    from merge_and_rebase.data.llm_calibration import _task_docs

    eval_samples = rt._calibration().eval_samples
    if len(eval_samples) != 1:
        raise ValueError(f"expected exactly one held-out harness task, got {sorted(eval_samples)}")
    ((task_name, indices),) = eval_samples.items()
    task = TaskManager().load_task_or_group([task_name])[task_name]
    docs = _task_docs(task)
    out = []
    for i in indices:
        doc = docs[i]
        out.append({"task": task_name, "doc_index": int(i), "key": doc.get("key"), "text": task.doc_to_text(doc)})
    return out


def build_llm_inputs(cfg: dict, device: str, context_sink: dict[str, Any] | None = None) -> Iterator[DiagnosticInputs]:
    """``context_sink`` (default ``None``: behaviour unchanged) is refilled with ``runtime``, ``run_cfg``,
    ``ref_summary``, ``build_cfg`` (name -> TextBuildConfig as the run builds it), ``eval_samples`` and the rendered
    held-out ``heldout_docs``. With ``context_sink["context_only"]`` true the generator returns right after.
    """
    # Imported here: llm_rebase pulls in lm-eval/transformers machinery the vision adapter never needs.
    from merge_and_rebase.eval.llm_rebase.context import _ariadne_params, build_runtime
    from merge_and_rebase.eval.llm_rebase.run_config import resolve_llm_method
    from merge_and_rebase.eval.llm_rebase.stages import build_task_models

    ref_summary_path = str(cfg["reference_summary"])
    ref_config_path = str(cfg["reference_config"])
    ref_summary = json.loads(Path(ref_summary_path).read_text())
    run_cfg: dict[str, Any] = json.loads(Path(ref_config_path).read_text())
    embedded = ((ref_summary.get("run_logging") or {}).get("metadata") or {}).get("resolved_config")
    if embedded is not None:
        for key in (
            "ariadne_params",
            "calibration_batch_size",
            "calibration_max_length",
            "tuned_bodies",
            "dtype",
            "source_model_name_or_path",
            "target_model_name_or_path",
            "harness_tasks",
            "seed",
        ):
            if embedded.get(key) != run_cfg.get(key):
                raise ValueError(f"reference_config disagrees with the summary's resolved config on {key!r}")
    alpha_star = float(json.loads(Path(str(cfg["alpha_star_summary"])).read_text())["best_alpha"])

    # The run's own config, minus everything that would evaluate or write: the fit inputs are unchanged by these.
    # `device` is the diagnostic's device (the run used cfg["device"]); `eval_before_rebase` only gates a harness
    # eval closure inside build_runtime; `save_merged`/`logging` are output-side.
    run_cfg = copy.deepcopy(run_cfg)
    run_cfg["device"] = device
    run_cfg["eval_before_rebase"] = False
    run_cfg.pop("save_merged", None)
    if str(run_cfg.get("method")) != "ariadne":
        raise ValueError("reference_config is not an ariadne run")

    # -- exactly llm_rebase/cli.py main(): method resolution and the two TextBuildConfigs --------------------------
    method_name, method, method_params = resolve_llm_method(run_cfg)
    model_arch = str(run_cfg.get("model_arch", "auto"))
    model_kind = str(run_cfg.get("model_kind", "causal_lm"))
    num_labels = int(run_cfg.get("num_labels", 3))
    dtype = run_cfg.get("dtype", None)

    def _build_cfg(name: str) -> TextBuildConfig:
        return TextBuildConfig(
            model_name_or_path=str(name),
            model_arch=model_arch,
            device=device,
            dtype=dtype,
            model_kind=model_kind,
            num_labels=num_labels,
            trust_remote_code=bool(run_cfg.get("trust_remote_code", False)),
            use_fast_tokenizer=bool(run_cfg.get("use_fast_tokenizer", True)),
        )

    rt = build_runtime(
        run_cfg,
        method_name=method_name,
        method=method,
        method_params=method_params,
        source_build_cfg=_build_cfg(run_cfg["source_model_name_or_path"]),
        target_build_cfg=_build_cfg(run_cfg["target_model_name_or_path"]),
        device=device,
        num_labels=num_labels,
    )
    config = rt.resolved.ariadne_cfg
    if config is None:
        raise ValueError("resolved run has no Ariadne config")
    # The config the run resolved: resolve_ariadne_decoder_config(_ariadne_params(cfg), family) in build_runtime.
    if config.activation_storage != "streaming":
        raise ValueError(f"expected activation_storage='streaming', got {config.activation_storage!r}")
    if tuple(config.components) != ("mlp.c_proj",):
        raise ValueError(f"expected a single mlp.c_proj component, got {config.components!r}")
    if config.ridge_estimator != "empirical_bayes":
        raise ValueError(f"expected ridge_estimator='empirical_bayes', got {config.ridge_estimator!r}")
    if _ariadne_params(run_cfg).get("seed", config.seed) != config.seed:
        raise ValueError("resolved Ariadne seed differs from ariadne_params.seed")
    family_adapter = rt.family_adapter
    layout = _layout_for(family_adapter)

    env = StageEnv(
        resolved=rt.resolved,
        plan=rt.plan,
        cfg=run_cfg,
        device=device,
        source_base_sd=rt.source_base_sd,
        target_base_sd=rt.target_base_sd,
        run_logger=None,
        runtime=rt,
    )
    wanted = cfg.get("tasks", "all")
    task_names = list(rt.task_contexts)
    if wanted != "all":
        wanted = [wanted] if isinstance(wanted, str) else list(wanted)
        missing = [t for t in wanted if t not in rt.task_contexts]
        if missing:
            raise ValueError(f"tasks {missing} not in the reference run's tasks {task_names}")
        task_names = wanted

    if context_sink is not None:
        context_only = bool(context_sink.get("context_only"))
        context_sink.clear()
        context_sink.update(
            context_only=context_only,
            runtime=rt,
            run_cfg=run_cfg,
            ref_summary=ref_summary,
            build_cfg=_build_cfg,
            eval_samples=rt._calibration().eval_samples,
            heldout_docs=_heldout_docs(rt),
        )
        if context_only:
            return

    # -- DirectFitStage.run, step by step ---------------------------------------------------------------------------
    materialized: list[str] = []
    if config.missing_bias == "materialize":
        materialized = list(
            materialize_missing_projection_biases(
                rt.target_llm.model, rt.target_base_sd, family_adapter=family_adapter, components=config.components
            )
        )
    texts = rt._calibration().texts
    source_loader = build_text_calibration_loader(
        tokenizer=rt.source_llm.tokenizer, texts=texts, batch_size=rt.calib_batch_size, max_length=rt.calib_max_length
    )
    target_loader = build_text_calibration_loader(
        tokenizer=rt.target_llm.tokenizer, texts=texts, batch_size=rt.calib_batch_size, max_length=rt.calib_max_length
    )
    pairing = apply_depth_pairing_override(
        DiscreteLayerPairing.compute(int(rt.source_depth), int(rt.target_depth)), config.depth_pairing
    )

    def component_key(j: int) -> str:
        return layout.component_key(int(j), "mlp.c_proj", prefixed=True)

    source_component_key = component_key  # the same decoder layout/adapter names both models' blocks

    # -- calibration provenance: recomputed corpus against the reference summary's ---------------------------------
    ours = rt._calibration().provenance()
    ref_cal = ref_summary.get("calibration_provenance") or {}
    cal_compare = {
        k: {"reference": ref_cal.get(k), "recomputed": ours.get(k), "match": ref_cal.get(k) == ours.get(k)}
        for k in ("corpus_sha256", "holdout_sha256", "n_sequences", "source")
    }

    # -- generic (off-task) target batches --------------------------------------------------------------------------
    n_generic = int(cfg.get("generic_num_batches", 25))
    generic_texts, generic_prov = _generic_texts(
        str(cfg.get("generic_dataset", "hellaswag")),
        n_generic * int(rt.calib_batch_size),
        int(cfg.get("generic_seed", 0)),
    )
    generic_loader = build_text_calibration_loader(
        tokenizer=rt.target_llm.tokenizer,
        texts=generic_texts,
        batch_size=rt.calib_batch_size,
        max_length=rt.calib_max_length,
    )
    generic_batches = []
    for batch in generic_loader:
        _, tgt, _, _ = _attach_row_masks(batch, batch, family_adapter)
        generic_batches.append(tgt)
    if len(generic_batches) != n_generic:
        raise ValueError(f"built {len(generic_batches)} generic batches, expected {n_generic}")
    tok = rt.target_llm.tokenizer
    generic_prov["tokenizer"] = {
        "padding_side": tok.padding_side,
        "bos_token": tok.bos_token,
        "pad_token": tok.pad_token,
        "max_length": int(rt.calib_max_length),
        "batch_size": int(rt.calib_batch_size),
    }

    target_depth = int(rt.target_depth)
    ref_keys = [k for j in range(target_depth) for k in (component_key(j), _bias_key(component_key(j)))]

    for task in task_names:
        models = build_task_models(env, task)
        if models is None:
            raise RuntimeError("build_task_models returned None for an Ariadne run")
        source_base_model = models.source_base.to(device)
        source_ft_model = models.source_ft.to(device)
        rt.target_llm.model.to(device)

        # Lazily, once: the 7.1 GB fp32 merged state is dropped as soon as the 56 tensors are copied out.
        reference_tv = _load_reference_task_vector(str(cfg["reference_merged_state"]), rt.target_base_sd, ref_keys)

        provenance = {
            "reference_summary": ref_summary_path,
            "reference_summary_sha256": _sha256_file(ref_summary_path),
            "reference_config": ref_config_path,
            "reference_config_sha256": _sha256_file(ref_config_path),
            "reference_merged_state": str(cfg["reference_merged_state"]),
            "alpha_star_summary": str(cfg["alpha_star_summary"]),
            "reference_code_fingerprint": ((ref_summary.get("run_logging") or {}).get("metadata") or {}).get(
                "code_fingerprint"
            ),
            "calibration": {"comparison": cal_compare, "recomputed": ours, "reference": ref_cal},
            "calibration_matches_reference": all(v["match"] for v in cal_compare.values()),
            "generic": generic_prov,
            "materialized_bias_keys": materialized,
            "pairing": list(pairing.pairing),
            "ariadne_seed": config.seed,
            "run_seed": int(run_cfg.get("seed", 0)),
        }
        yield DiagnosticInputs(
            family="llm",
            task=task,
            device=device,
            config=config,
            pairing=pairing,
            source_base_model=source_base_model,
            source_ft_model=source_ft_model,
            target_model=rt.target_llm.model,
            target_base_state=rt.target_base_sd,
            source_loader=source_loader,
            target_loader=target_loader,
            generic_target_batches=generic_batches,
            token_index=llm_token_index,
            component_key=component_key,
            source_component_key=source_component_key,
            family_adapter=family_adapter,
            alpha_star=alpha_star,
            reference_task_vector=reference_tv,
            reference_task_vector_tolerance="tolerance",
            reference_rows=None,
            provenance=provenance,
        )
