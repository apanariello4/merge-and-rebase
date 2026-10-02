"""Alpha search loops: lm-harness and NLI evaluator backends (moved out of cli.main, behaviour unchanged)."""

from __future__ import annotations

from typing import Any

from ...hyperparam_search import SearchEvaluation, describe_candidate
from ...io.ckpt import load_into_model
from ...merge.runtime import apply_delta
from .common import default_prompt_for_task, inject_task_head, normalized_acc


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
