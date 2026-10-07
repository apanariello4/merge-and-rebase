"""Alpha search loops: lm-harness and NLI evaluator backends (moved out of cli.main, behaviour unchanged)."""

from __future__ import annotations

from typing import Any

from ...data.text_loaders import NLITaskData, NLITokenizedData, build_nli_task_data, build_nli_tokenized_loader
from ...hyperparam_search import SearchEvaluation, build_search_planner, describe_candidate, summarize_search_results
from ...io.ckpt import load_into_model
from ...merge.runtime import apply_delta
from ..print_utils import pretty_print_task_accuracies
from .artifacts import save_merged_state, save_transported_task_vector
from .common import (
    default_prompt_for_task,
    head_class_ids_for_task,
    inject_task_head,
    load_task_heads,
    normalized_acc,
    resolve_fine_tuned_acc,
    resolve_task_mask_class,
    to_unit_acc,
)
from .summary import RunRecord


def harness_alpha_search(
    *,
    search_planner: Any,
    merged_delta: dict[str, Any],
    target_base_sd: dict[str, Any],
    target_llm: Any,
    harness_tasks_resolved: Any,
    device: Any,
    harness_num_fewshot: Any,
    harness_batch_size: Any,
    harness_limit: Any,
    harness_samples: Any,
    harness_render: Any = None,
) -> tuple[SearchEvaluation, dict[float, dict[str, float]], list[SearchEvaluation]]:
    from .harness import run as run_harness
    from .harness import score_by_task

    best_harness_eval: SearchEvaluation | None = None
    harness_results_by_alpha: dict[float, dict[str, float]] = {}
    harness_search_results: list[SearchEvaluation] = []

    while True:
        batch = search_planner.next_batch()
        if batch is None:
            break
        batch_results: list[SearchEvaluation] = []

        for candidate in batch:
            alpha = float(candidate.alpha)
            scaled = {k: v * alpha for k, v in merged_delta.items()}
            merged_sd = apply_delta(target_base_sd, scaled)
            load_into_model(target_llm.model, merged_sd, strict=False)

            print(f"\nEvaluating with lm-harness (alpha={alpha:.3f})...")
            harness_results = run_harness(
                tasks=list(harness_tasks_resolved),
                model=target_llm.model,
                tokenizer=target_llm.tokenizer,
                device=device,
                num_fewshot=harness_num_fewshot,
                batch_size=harness_batch_size,
                limit=harness_limit,
                samples=harness_samples,
                **(harness_render or {}),
            )
            for task_name, acc in harness_results.items():
                print(f"  {task_name}: {acc:.4f}")

            score = score_by_task(harness_results, list(harness_tasks_resolved))
            result = SearchEvaluation(
                candidate=candidate,
                score=float(score),
                avg_acc=float(score),
                avg_norm_acc=0.0,
                per_task_acc=[float(v) for v in harness_results.values()],
                per_task_norm_acc=[],
            )
            batch_results.append(result)
            harness_search_results.append(result)
            harness_results_by_alpha[alpha] = harness_results

            if best_harness_eval is None or result.score > best_harness_eval.score:
                best_harness_eval = result

            print(f"  alpha={alpha:.3f}  avg_score={score:.6f}")
            del merged_sd

        search_planner.observe(batch_results)

    if best_harness_eval is None:
        raise RuntimeError("Harness alpha search produced no results.")

    if len(harness_search_results) > 1:
        print("\n=== Harness alpha search summary ===")
        for r in harness_search_results:
            print(f"{describe_candidate(r.candidate)}  avg_score={r.avg_acc:.6f}")

    return best_harness_eval, harness_results_by_alpha, harness_search_results


def score_harness_test_slice(
    *,
    cfg: dict[str, Any],
    merged_delta: dict[str, Any],
    target_base_sd: dict[str, Any],
    target_llm: Any,
    harness_tasks_resolved: Any,
    device: Any,
    harness_num_fewshot: Any,
    harness_batch_size: Any,
    harness_samples: Any,
    best_alpha: float,
    harness_render: Any = None,
) -> tuple[dict[str, float] | None, dict[str, list[int]] | None]:
    from .harness import run as run_harness

    # Optional held-out test slice. The alpha search above selects on
    # `harness_samples`; selecting and reporting on the same documents
    # makes the reported number the maximum over the alpha grid on that
    # slice, which is biased upward and -- on a small slice -- by more
    # than the effects being compared. When `harness_test_samples` is
    # configured, the winning alpha is re-scored once on those disjoint
    # documents and that is the number to quote. Costs one extra pass,
    # not one per alpha.
    harness_test_results: dict[str, float] | None = None
    test_samples_cfg = cfg.get("harness_test_samples", None)
    if test_samples_cfg is not None:
        if not isinstance(test_samples_cfg, dict):
            raise ValueError("config['harness_test_samples'] must map task names to index lists.")
        test_samples = {}
        for task_name, indices in test_samples_cfg.items():
            if not isinstance(indices, list) or not all(isinstance(i, int) and i >= 0 for i in indices):
                raise ValueError(
                    "config['harness_test_samples'] values must be lists of non-negative indices."
                )
            test_samples[str(task_name)] = list(indices)
        overlap = {
            t: sorted(set(test_samples.get(t, ())) & set((harness_samples or {}).get(t, ())))
            for t in test_samples
        }
        leaking = {t: v for t, v in overlap.items() if v}
        if leaking:
            raise ValueError(
                "harness_test_samples overlaps the alpha-search slice for "
                f"{ {t: len(v) for t, v in leaking.items()} }; the reported number would be "
                "selected on documents it is scored on."
            )
        scaled = {k: v * best_alpha for k, v in merged_delta.items()}
        load_into_model(target_llm.model, apply_delta(target_base_sd, scaled), strict=False)
        print(f"\nScoring held-out test slice at alpha={best_alpha:.3f}...")
        harness_test_results = run_harness(
            tasks=list(harness_tasks_resolved),
            model=target_llm.model,
            tokenizer=target_llm.tokenizer,
            device=device,
            num_fewshot=harness_num_fewshot,
            batch_size=harness_batch_size,
            limit=None,
            samples=test_samples,
            **(harness_render or {}),
        )
        print("=== Harness results (held-out test slice) ===")
        for task_name, acc in harness_test_results.items():
            print(f"  {task_name}: {acc:.4f}")

    return harness_test_results, (test_samples if test_samples_cfg is not None else None)


def nli_alpha_search(
    *,
    search_planner: Any,
    merged_delta: dict[str, Any],
    target_base_sd: dict[str, Any],
    target_llm: Any,
    task_data: list[Any],
    tokenized_task_data: list[Any],
    task_heads: Any,
    eval_mode: str,
    head_key_pattern: Any,
    user_prompt_template: Any,
    max_prompt_tokens: Any,
    print_every: Any,
    external_ref_acc: Any,
    device: Any,
) -> tuple[SearchEvaluation, list[SearchEvaluation]]:
    best_result: SearchEvaluation | None = None
    search_results: list[SearchEvaluation] = []
    alpha_to_task_accs: dict[float, list[float]] = {}
    alpha_to_task_norm_accs: dict[float, list[float]] = {}

    while True:
        batch = search_planner.next_batch()
        if batch is None:
            break
        batch_results: list[SearchEvaluation] = []

        for candidate in batch:
            alpha = float(candidate.alpha)

            scaled = {k: v * alpha for k, v in merged_delta.items()}
            merged_sd = apply_delta(target_base_sd, scaled)
            load_into_model(target_llm.model, merged_sd, strict=False)

            accs: list[float] = []
            norm_accs: list[float] = []
            for i, td in enumerate(task_data):
                if eval_mode == "head_logits" and task_heads is not None:
                    tk = tokenized_task_data[i]
                    inject_task_head(
                        model=target_llm.model,
                        task=td.task,
                        task_heads=task_heads,
                        head_key_pattern=head_key_pattern,
                        head_class_ids=list(tk.meta.get("head_class_ids", [])),
                    )
                    acc = target_llm.sequence_classification_accuracy(
                        tk.loader,
                        device=device,
                        mask_class=tk.mask_class,
                        print_every=print_every,
                    )
                else:
                    tpl = user_prompt_template if user_prompt_template else default_prompt_for_task(td)
                    acc = target_llm.nli_accuracy(
                        examples=td.examples,
                        label_texts=td.label_texts,
                        prompt_template=tpl,
                        device=device,
                        max_prompt_tokens=max_prompt_tokens,
                        print_every=print_every,
                    )
                accs.append(acc)
                if external_ref_acc is not None and td.task in external_ref_acc:
                    n = normalized_acc(acc, external_ref_acc[td.task])
                    norm_accs.append(n)
                    print(f"  {td.task}: acc={acc:.6f}  norm_acc={n:.3f}")
                else:
                    print(f"  {td.task}: acc={acc:.6f}")

            avg_acc = sum(accs) / max(1, len(accs))
            avg_norm_acc = sum(norm_accs) / max(1, len(norm_accs)) if norm_accs else 0.0
            score = avg_norm_acc if norm_accs else avg_acc
            result = SearchEvaluation(
                candidate=candidate,
                score=float(score),
                avg_acc=float(avg_acc),
                avg_norm_acc=float(avg_norm_acc),
                per_task_acc=[float(v) for v in accs],
                per_task_norm_acc=[float(v) for v in norm_accs],
            )
            batch_results.append(result)
            search_results.append(result)
            alpha_to_task_accs[alpha] = [float(v) for v in accs]
            alpha_to_task_norm_accs[alpha] = [float(v) for v in norm_accs]

            if best_result is None or result.score > best_result.score:
                best_result = result

            print(f"  alpha={alpha:.2f}  avg_acc={avg_acc:.6f}  avg_norm_acc={avg_norm_acc:.3f}" if norm_accs else f"  alpha={alpha:.2f}  avg_acc={avg_acc:.6f}")

            del merged_sd

        search_planner.observe(batch_results)

    if best_result is None:
        raise RuntimeError("Alpha search produced no results.")

    print("\n=== Alpha search summary ===")
    for r in search_results:
        if r.per_task_norm_acc:
            print(
                f"{describe_candidate(r.candidate)}  "
                f"avg_acc={r.avg_acc:.6f}  avg_norm_acc={r.avg_norm_acc:.3f}"
            )
        else:
            print(f"{describe_candidate(r.candidate)}  avg_acc={r.avg_acc:.6f}")

    return best_result, search_results


def run_alpha_search(rt: Any, outputs: Any) -> RunRecord:
    """Dispatch the evaluation backend (lm-harness or NLI), search alpha, score, save; return the summary record."""
    cfg = rt.cfg
    method_name = rt.method_name
    method_params = rt.method_params
    target_llm = rt.target_llm
    target_base_sd = rt.target_base_sd
    device = rt.device
    harness_tasks_resolved = rt.harness_tasks_resolved
    harness_num_fewshot = rt.harness_num_fewshot
    harness_batch_size = rt.harness_batch_size
    harness_limit = rt.harness_limit
    harness_samples = rt.harness_samples
    harness_render = rt.harness_render
    is_harness_only = rt.is_harness_only
    ignored_block_extension_fields = rt.ignored_block_extension_fields
    _calibration_provenance = rt._calibration_provenance
    run_block_extension_prestep = rt.run_block_extension_prestep
    tasks = rt.tasks
    suite_name = rt.suite_name
    eval_mode = rt.eval_mode
    head_key_pattern = rt.head_key_pattern
    task_heads_path = rt.task_heads_path
    num_labels = rt.num_labels
    _baseline_summary = rt._baseline_summary
    merged_delta = outputs.merged_delta
    delta_stats = outputs.delta_stats
    task_vector_report = outputs.task_vector_report

    search_planner = build_search_planner(
        cfg=cfg, base_method_params=method_params
    )

    # ---- Dispatch evaluation backend ----
    if is_harness_only or harness_tasks_resolved:

        baseline_harness_results = _baseline_summary()

        best_harness_eval, harness_results_by_alpha, harness_search_results = harness_alpha_search(
            search_planner=search_planner,
            merged_delta=merged_delta,
            target_base_sd=target_base_sd,
            target_llm=target_llm,
            harness_tasks_resolved=harness_tasks_resolved,
            device=device,
            harness_num_fewshot=harness_num_fewshot,
            harness_batch_size=harness_batch_size,
            harness_limit=harness_limit,
            harness_samples=harness_samples,
            harness_render=harness_render,
        )

        best_alpha = float(best_harness_eval.candidate.alpha)
        best_harness_results = harness_results_by_alpha[best_alpha]
        print(f"\nBest alpha={best_alpha:.3f} -> avg_score={best_harness_eval.avg_acc:.6f}")
        print("\n=== Harness results (best alpha) ===")
        for task_name, acc in best_harness_results.items():
            print(f"  {task_name}: {acc:.4f}")

        harness_test_results, test_samples = score_harness_test_slice(
            cfg=cfg,
            merged_delta=merged_delta,
            target_base_sd=target_base_sd,
            target_llm=target_llm,
            harness_tasks_resolved=harness_tasks_resolved,
            device=device,
            harness_num_fewshot=harness_num_fewshot,
            harness_batch_size=harness_batch_size,
            harness_samples=harness_samples,
            best_alpha=best_alpha,
            harness_render=harness_render,
        )

        if cfg.get("save_merged", None) is not None:
            save_merged_state(
                cfg["save_merged"], merged_delta, best_alpha, target_base_sd, message="Saved rebased state to"
            )
        transported_tv = save_transported_task_vector(
            cfg,
            merged_delta,
            method_name=method_name,
            best_alpha=best_alpha,
            alpha_curve={
                "search_results": summarize_search_results(harness_search_results),
                "harness_results_by_alpha": {f"{a:g}": r for a, r in sorted(harness_results_by_alpha.items())},
            },
        )

        return RunRecord(
            "harness",
            dict(
                ignored_block_extension_fields=ignored_block_extension_fields,
                calibration_provenance=_calibration_provenance(),
                method_name=method_name,
                best_alpha=best_alpha,
                best_harness_results=best_harness_results,
                harness_test_results=harness_test_results,
                test_samples=test_samples,
                baseline_harness_results=baseline_harness_results,
                run_block_extension_prestep=run_block_extension_prestep,
                harness_results_by_alpha=harness_results_by_alpha,
                delta_stats=delta_stats,
                task_vector_report=task_vector_report,
                search_planner=search_planner,
                harness_search_results=harness_search_results,
                saved_merged_path=cfg.get("save_merged"),
                transported_tv=transported_tv,
            ),
        )

    # ---- NLI eval path (existing) ----
    task_heads: dict[str, Any] | None = None
    if task_heads_path is not None:
        task_heads = load_task_heads(str(task_heads_path))

    user_prompt_template = cfg.get("prompt_template", None)
    split = str(cfg.get("split", "validation"))
    max_samples_per_task = cfg.get("max_samples_per_task", None)
    if max_samples_per_task is not None:
        max_samples_per_task = int(max_samples_per_task)
    max_prompt_tokens = cfg.get("max_prompt_tokens", None)
    if max_prompt_tokens is not None:
        max_prompt_tokens = int(max_prompt_tokens)
    print_every = cfg.get("print_every", None)
    if print_every is not None:
        print_every = int(print_every)

    task_data: list[NLITaskData] = []
    for t in tasks:
        td = build_nli_task_data(task=t, split=split, max_samples=max_samples_per_task)
        task_data.append(td)
        print(f"Loaded task {t}: {td.meta}")

    external_ref_acc = resolve_fine_tuned_acc(cfg=cfg, tasks=tasks)
    if external_ref_acc is not None:
        print(f"External ref accs: {external_ref_acc}")

    tokenized_task_data: list[NLITokenizedData] = []
    if eval_mode == "head_logits" and task_heads is not None:
        batch_size = int(cfg.get("batch_size", 8))
        num_workers = int(cfg.get("num_workers", 0))
        max_length = int(cfg.get("max_length", 512))
        task_mask_class = resolve_task_mask_class(cfg.get("task_mask_class", {}))
        head_num_labels = int(getattr(target_llm.model.config, "num_labels", num_labels))

        for td in task_data:
            masked_class = task_mask_class.get(td.task, None)
            class_ids = head_class_ids_for_task(
                task=td.task,
                task_num_labels=len(td.labels),
                head_num_labels=head_num_labels,
                masked_class=masked_class,
            )
            tk = build_nli_tokenized_loader(
                task_data=td,
                tokenizer=target_llm.tokenizer,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                head_class_ids=class_ids,
            )
            tokenized_task_data.append(tk)
            print(f"Tokenized {td.task}: {tk.meta}")

    best_result, search_results = nli_alpha_search(
        search_planner=search_planner,
        merged_delta=merged_delta,
        target_base_sd=target_base_sd,
        target_llm=target_llm,
        task_data=task_data,
        tokenized_task_data=tokenized_task_data,
        task_heads=task_heads,
        eval_mode=eval_mode,
        head_key_pattern=head_key_pattern,
        user_prompt_template=user_prompt_template,
        max_prompt_tokens=max_prompt_tokens,
        print_every=print_every,
        external_ref_acc=external_ref_acc,
        device=device,
    )

    best_alpha = float(best_result.candidate.alpha)
    best_vals = list(best_result.per_task_acc)
    print(f"\nBest alpha={best_alpha:.2f} -> avg_acc={best_result.avg_acc:.6f}")

    if external_ref_acc is not None:
        per_task_rows = [{"task": td.task} for td in task_data]
        single_accs = [
            to_unit_acc(external_ref_acc[td.task]) if td.task in external_ref_acc else 0.0
            for td in task_data
        ]
        norm_ratio = [
            (best_vals[i] / single_accs[i]) if single_accs[i] > 0 else 0.0
            for i in range(len(best_vals))
        ]
        pretty_print_task_accuracies(
            suite_name or "nli6",
            method_name,
            "full",
            per_task_rows,
            best_vals,
            norm_ratio,
            single_accs=single_accs,
        )

    if cfg.get("save_merged", None) is not None:
        save_merged_state(
            cfg["save_merged"], merged_delta, best_alpha, target_base_sd, message="Saved best-alpha rebased state to"
        )
    transported_tv = save_transported_task_vector(
        cfg,
        merged_delta,
        method_name=method_name,
        best_alpha=best_alpha,
        alpha_curve=summarize_search_results(search_results),
    )

    return RunRecord(
        "nli",
        dict(
            ignored_block_extension_fields=ignored_block_extension_fields,
            calibration_provenance=_calibration_provenance(),
            method_name=method_name,
            best_alpha=best_alpha,
            task_data=task_data,
            delta_stats=delta_stats,
            task_vector_report=task_vector_report,
            search_planner=search_planner,
            search_results=search_results,
            best_vals=best_vals,
            saved_merged_path=cfg.get("save_merged"),
            transported_tv=transported_tv,
        ),
    )
