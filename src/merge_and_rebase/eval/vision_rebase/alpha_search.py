"""Evaluator and alpha search for the vision rebase entrypoint.

``TargetEvaluator`` scores deltas on the target model (it replaces the former ``main()`` closures);
``run_alpha_search`` runs the shared / per-task / hierarchical (pass 1 per-task premerge alphas, pass 2 global
merge alpha) sweeps with patience early stopping, the single-TV diagnostic and the test re-evaluation.
Bodies are the former inline blocks of ``main()``, moved verbatim; the only change is that the hierarchical pass 2
swaps the evaluator's rebased deltas explicitly (``with_rebased_deltas``) instead of rebinding a closure variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...io.ckpt import load_into_model
from ...merge.methods._common import axpy_state_dict
from ...utils.alpha_search import PerTaskAlphaTracker, average_scores
from ..rebase_metrics import normalized_accuracy_ratio
from ..utils import eval_task_top1
from .merge import _TRANSPORT_THEN_MERGE_MODES, MergePlan, _merge_direction, _scale_deltas_by


def _norm_acc(result_acc: float, baseline_acc: float) -> float:
    return normalized_accuracy_ratio(result_acc, baseline_acc)


def _average_defined(values: list[float]) -> float:
    defined = [float(v) for v in values if float(v) == float(v)]
    return average_scores(defined) if defined else float("nan")


class TargetEvaluator:
    """Loads ``target_base + alpha * delta`` into the target model and scores it on a task split.

    ``target_base_sd`` is read at construction (after the per-task loop, where TransFusion's once-only
    prepare may have swapped it). The zero-shot cache is shared between evaluators derived through
    :meth:`with_rebased_deltas`.
    """

    def __init__(
        self,
        *,
        clf_target: Any,
        per_task: list[dict[str, Any]],
        device: str,
        target_base_sd: dict[str, torch.Tensor],
        method: Any,
        transfusion_mode: bool,
        strict_load: bool,
        rebased_deltas: list[dict[str, torch.Tensor]],
        untransported_deltas: list[dict[str, torch.Tensor]],
        can_eval_untransported: list[bool],
        single_tv_deltas: list[dict[str, torch.Tensor]] | None,
        zeroshot_cache: dict[str, list[float]] | None = None,
    ) -> None:
        self.clf_target = clf_target
        self.per_task = per_task
        self.device = device
        self.target_base_sd = target_base_sd
        self.method = method
        self.transfusion_mode = transfusion_mode
        self.strict_load = strict_load
        self.rebased_deltas = rebased_deltas
        self.untransported_deltas = untransported_deltas
        self.can_eval_untransported = can_eval_untransported
        self.single_tv_deltas = single_tv_deltas
        self._zeroshot_cache: dict[str, list[float]] = {} if zeroshot_cache is None else zeroshot_cache

    @classmethod
    def from_plan(
        cls,
        env: Any,
        merge_plan: MergePlan,
        *,
        per_task: list[dict[str, Any]],
    ) -> TargetEvaluator:
        """Evaluator over the run's target model and the merge plan's deltas."""
        resolved = env.resolved
        return cls(
            clf_target=env.clf_target,
            per_task=per_task,
            device=env.device,
            target_base_sd=env.target_base_sd,
            method=resolved.method,
            transfusion_mode=resolved.transfusion_mode,
            strict_load=resolved.strict_load,
            rebased_deltas=merge_plan.rebased_deltas,
            untransported_deltas=merge_plan.untransported_deltas,
            can_eval_untransported=merge_plan.can_eval_untransported,
            single_tv_deltas=merge_plan.single_tv_deltas,
        )

    def with_rebased_deltas(self, rebased_deltas: list[dict[str, torch.Tensor]]) -> TargetEvaluator:
        return TargetEvaluator(
            clf_target=self.clf_target,
            per_task=self.per_task,
            device=self.device,
            target_base_sd=self.target_base_sd,
            method=self.method,
            transfusion_mode=self.transfusion_mode,
            strict_load=self.strict_load,
            rebased_deltas=rebased_deltas,
            untransported_deltas=self.untransported_deltas,
            can_eval_untransported=self.can_eval_untransported,
            single_tv_deltas=self.single_tv_deltas,
            zeroshot_cache=self._zeroshot_cache,
        )

    def eval_task(self, item: dict[str, Any], split: str) -> float:
        return float(
            eval_task_top1(
                clf=self.clf_target,
                loaders=item["loaders"],
                classnames=list(item["classnames"]),
                build_cfg_task=item["build_cfg_task"],
                device=self.device,
                split=split,
            )
        )

    def eval_all_tasks(self, split: str) -> list[float]:
        return [self.eval_task(item, split) for item in self.per_task]

    def load_into_target_model(self, sd: dict[str, torch.Tensor]) -> None:
        if self.transfusion_mode:
            self.method.load_into_target_visual(self.clf_target, sd, strict=False)
        else:
            load_into_model(self.clf_target.model, sd, strict=self.strict_load)

    def eval_zeroshot_all_tasks(self, split: str) -> list[float]:
        if split not in self._zeroshot_cache:
            self.load_into_target_model(self.target_base_sd)
            self._zeroshot_cache[split] = self.eval_all_tasks(split)
        return list(self._zeroshot_cache[split])

    def eval_baseline(self, split: str, idx: int, alpha: float) -> float:
        if self.can_eval_untransported[idx]:
            baseline_sd = axpy_state_dict(self.target_base_sd, self.untransported_deltas[idx], alpha=float(alpha))
            self.load_into_target_model(baseline_sd)
            del baseline_sd
            return self.eval_task(self.per_task[idx], split)
        return self.eval_zeroshot_all_tasks(split)[idx]

    def eval_baseline_indices(self, split: str, indices: list[int], alpha: float) -> dict[int, float]:
        return {idx: self.eval_baseline(split, idx, alpha) for idx in indices}

    def eval_rebased_indices(self, split: str, indices: list[int], alpha: float) -> dict[int, float]:
        out: dict[int, float] = {}
        for idx in indices:
            rebase_sd_task = axpy_state_dict(self.target_base_sd, self.rebased_deltas[idx], alpha=float(alpha))
            self.load_into_target_model(rebase_sd_task)
            del rebase_sd_task
            out[idx] = self.eval_task(self.per_task[idx], split)
        return out

    def eval_single_tv_indices(self, split: str, indices: list[int], alpha: float) -> dict[int, float]:
        if self.single_tv_deltas is None:
            raise RuntimeError("Single-TV diagnostic requires merge_mode='rebase_then_merge'.")
        out: dict[int, float] = {}
        for idx in indices:
            single_sd = axpy_state_dict(self.target_base_sd, self.single_tv_deltas[idx], alpha=float(alpha))
            self.load_into_target_model(single_sd)
            del single_sd
            out[idx] = self.eval_task(self.per_task[idx], split)
        return out


@dataclass
class AlphaSearchSpec:
    """Everything the alpha search reads from the resolved config and the run."""

    per_task: list[dict[str, Any]]
    alphas: list[float]
    selection: str
    patience: int
    search_split: str
    global_alpha_search: bool
    merge_mode: str
    merge_method_name: str
    merge_params: dict[str, Any]
    merge_weights: list[float]
    method_label: str
    run_logger: Any

    @classmethod
    def from_run(
        cls,
        resolved: Any,
        *,
        per_task: list[dict[str, Any]],
        merge_weights: list[float],
        method_label: str,
        run_logger: Any,
    ) -> AlphaSearchSpec:
        return cls(
            per_task=per_task,
            alphas=resolved.alpha.alphas,
            selection=resolved.alpha.selection,
            patience=resolved.alpha.patience,
            search_split=resolved.alpha.search_split,
            global_alpha_search=resolved.merge.global_alpha_search,
            merge_mode=resolved.merge.mode,
            merge_method_name=resolved.merge.method_name,
            merge_params=resolved.merge.params,
            merge_weights=merge_weights,
            method_label=method_label,
            run_logger=run_logger,
        )


@dataclass
class AlphaResult:
    """Outputs of :func:`run_alpha_search` (the summary fields keep their legacy names)."""

    baseline_label: str
    result_label: str
    hierarchical: bool
    best_alpha: float
    best_baseline_alpha: float
    selected_alpha_by_task: list[float]
    selected_baseline_alpha_by_task: list[float]
    rebase_test_accs: list[float]
    baseline_test_accs: list[float]
    norm_accs: list[float]
    per_task_premerge_alphas: list[float] | None
    hierarchical_premerge_alpha_curve: list[dict[str, Any]] | None
    global_alpha_curve: list[dict[str, Any]] | None
    selected_validation_results: dict[str, Any] | None
    single_tv_test_accs: list[float] | None
    single_tv_alpha_protocol: str
    single_tv_val_best_alpha: list[float] | None
    single_tv_val_best_acc: list[float] | None
    #: Deltas of the final evaluator (the merged direction after a hierarchical pass 2).
    rebased_deltas: list[dict[str, torch.Tensor]]


def run_alpha_search(spec: AlphaSearchSpec, evaluator: TargetEvaluator, merge_plan: MergePlan) -> AlphaResult:
    per_task = spec.per_task
    alphas = spec.alphas
    alpha_selection = spec.selection
    alpha_patience = spec.patience
    alpha_search_split = spec.search_split
    global_alpha_search = spec.global_alpha_search
    merge_mode = spec.merge_mode
    merge_method_name = spec.merge_method_name
    merge_params = spec.merge_params
    merge_weights = spec.merge_weights
    method_label = spec.method_label
    run_logger = spec.run_logger
    can_eval_untransported_by_task = merge_plan.can_eval_untransported
    single_tv_deltas_for_diagnostic = merge_plan.single_tv_deltas
    ev = evaluator

    if all(can_eval_untransported_by_task):
        baseline_label = "untransported"
    elif any(can_eval_untransported_by_task):
        baseline_label = "mixed_baseline"
    else:
        baseline_label = "target_zeroshot"
    result_label = "rebased"
    task_col = max(max((len(str(item["task"])) for item in per_task), default=4), len("task"), len("avg"))
    metric_col = max(12, len(baseline_label) + 2, len(result_label) + 2, len("norm") + 2)

    if baseline_label == "untransported":
        print("Using untransported baseline evaluation for all tasks.")
    elif baseline_label == "mixed_baseline":
        print("Using mixed baseline evaluation: untransported where compatible, target zeroshot otherwise.")
    else:
        print("Using target zeroshot baseline for all tasks.")

    hierarchical = bool(merge_mode in _TRANSPORT_THEN_MERGE_MODES and alpha_selection == "per_task")
    single_tv_diagnostic_enabled = single_tv_deltas_for_diagnostic is not None
    single_tv_val_best_acc: list[float] | None = (
        [float("-inf")] * len(per_task) if single_tv_diagnostic_enabled else None
    )
    single_tv_val_best_alpha: list[float] | None = (
        [float(alphas[0])] * len(per_task) if single_tv_diagnostic_enabled else None
    )
    single_tv_alpha_protocol = "per_task_premerge_alpha" if hierarchical else "single_tv_validation_oracle"
    per_task_premerge_alphas: list[float] | None = None
    hierarchical_premerge_alpha_curve: list[dict[str, Any]] | None = None
    global_alpha_curve: list[dict[str, Any]] | None = None
    selected_validation_results: dict[str, Any] | None = None

    if alpha_selection == "shared" or hierarchical:
        if hierarchical:
            # ---------------- PASS 1: per-task alpha on individual deltas ----------------
            hierarchical_premerge_alpha_curve = []
            tracker = PerTaskAlphaTracker(
                task_names=[str(item["task"]) for item in per_task],
                initial_alpha=float(alphas[0]),
                patience=alpha_patience,
            )
            # Merge-mode baselines are the alpha-independent target zero-shot:
            # pre-seed the secondary stream inactive for every task.
            for idx in range(len(per_task)):
                baseline_val = ev.eval_baseline(alpha_search_split, idx, 0.0)
                tracker.best_secondary_alpha[idx] = 0.0
                tracker.best_secondary_acc[idx] = baseline_val
                tracker.secondary_active[idx] = False

            for alpha in alphas:
                eval_indices = tracker.eval_active_indices()
                if not eval_indices:
                    print("\nAll tasks have early-stopped; ending hierarchical pass-1 alpha sweep.")
                    break

                primary_indices = set(tracker.primary_active_indices())
                print(
                    f"\n=== alpha {alpha:.3f} — {method_label} "
                    f"(split: {alpha_search_split}, mode: per_task pass 1/2, hierarchical) ==="
                )

                baseline_by_idx = ev.eval_baseline_indices(alpha_search_split, eval_indices, float(alpha))
                rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
                rebase_by_idx_active = ev.eval_rebased_indices(alpha_search_split, rebase_eval_indices, float(alpha))
                rebase_by_idx: dict[int, float] = {}
                for idx in eval_indices:
                    rebase_by_idx[idx] = rebase_by_idx_active[idx] if idx in primary_indices else float("-inf")
                baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
                rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

                for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                    task_name = per_task[idx]["task"]
                    if idx in primary_indices:
                        display_rebase = rebase_acc
                        marker = " "
                    else:
                        frozen = float(tracker.best_primary_acc[idx])
                        display_rebase = frozen if frozen != float("-inf") else 0.0
                        marker = "*"
                    norm = _norm_acc(display_rebase, baseline_acc)
                    print(
                        f" {marker}{task_name}: {baseline_label}={baseline_acc:.6f}  "
                        f"per-task={display_rebase:.6f}  norm={norm:.6f}"
                    )

                hierarchical_premerge_alpha_curve.append(
                    {
                        "alpha": float(alpha),
                        "per_task_rebased": {
                            per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in rebase_eval_indices
                        },
                        "per_task_baseline": {
                            per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in eval_indices
                        },
                    }
                )
                run_logger.log_event(
                    "hierarchical_premerge_alpha_eval_end",
                    metrics={"alpha/value": float(alpha)},
                    context=hierarchical_premerge_alpha_curve[-1],
                )

                stopped_primary, _ = tracker.update(
                    alpha=float(alpha),
                    indices=eval_indices,
                    primary_accs=rebase_accs,
                    secondary_accs=baseline_accs,
                )
                if stopped_primary:
                    stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                    print(f"  Early-stopping per-task alphas at alpha={alpha:.3f}: {stopped_names}")

            per_task_premerge_alphas = [float(tracker.best_primary_alpha[idx]) for idx in range(len(per_task))]
            single_tv_val_best_acc = [float(tracker.best_primary_acc[idx]) for idx in range(len(per_task))]
            single_tv_val_best_alpha = list(per_task_premerge_alphas)
            print("\n=== Hierarchical pass-1 summary (per-task alphas) ===")
            for item, a in zip(per_task, per_task_premerge_alphas, strict=True):
                print(f"  {item['task']}: premerge_alpha={a:.3f}")
            run_logger.log_event(
                "hierarchical_pass1_end",
                metrics={
                    "hierarchical/avg_premerge_alpha": float(
                        sum(per_task_premerge_alphas) / max(1, len(per_task_premerge_alphas))
                    )
                },
                context={
                    "per_task_premerge_alphas": {
                        item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                    }
                },
            )

            # ---------------- PASS 2: scale by per-task alphas, compose once ----------------
            scaled_input = _scale_deltas_by(single_tv_deltas_for_diagnostic, per_task_premerge_alphas)
            merged_direction = _merge_direction(
                base_sd=ev.target_base_sd,
                deltas=scaled_input,
                merge_method_name=merge_method_name,
                weights=merge_weights,
                merge_params=merge_params,
            )
            # The second pass evaluates the merged direction: explicit new evaluator state, not a rebinding.
            ev = ev.with_rebased_deltas([merged_direction] * len(per_task))
            print(
                f"Hierarchical merge ({merge_method_name}): composed {len(scaled_input)} scaled deltas "
                f"-> merged direction with {len(merged_direction)} params"
            )
            run_logger.log_event(
                "merge_composition_end",
                metrics={"merge/param_count": float(len(merged_direction))},
                context={
                    "mode": merge_mode,
                    "merge_method": merge_method_name,
                    "merge_params": merge_params,
                    "n_tasks": len(scaled_input),
                    "per_task_premerge_alphas": {
                        item["task"]: float(per_task_premerge_alphas[i]) for i, item in enumerate(per_task)
                    },
                },
            )

        sweep_alphas = list(alphas)
        if hierarchical and not global_alpha_search:
            sweep_alphas = [1.0]
            print("\nglobal_alpha_search=false: evaluating the merged model at gamma=1.0 only.")
        sweep_positive_alphas = [float(a) for a in sweep_alphas if float(a) > 0.0]
        sweep_mode_label = "global (gamma)" if hierarchical else "shared"

        best_rebase_avg = float("-inf")
        best_baseline_avg = float("-inf")
        best_alpha = float(sweep_alphas[0])
        # When every task's baseline is target_zeroshot (untransported infeasible),
        # the baseline is alpha-independent — keep best_baseline_alpha at 0.0 so
        # the summary does not report a spurious non-zero value.
        has_untransported = any(can_eval_untransported_by_task)
        best_baseline_alpha = (
            float(sweep_positive_alphas[0] if sweep_positive_alphas else sweep_alphas[0]) if has_untransported else 0.0
        )
        sweep_results: list[dict[str, Any]] = []
        shared_bad_steps = 0

        for alpha in sweep_alphas:
            print(
                f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: {sweep_mode_label}) ==="
            )

            idxs = list(range(len(per_task)))
            baseline_by_idx = ev.eval_baseline_indices(alpha_search_split, idxs, float(alpha))
            rebase_by_idx = ev.eval_rebased_indices(alpha_search_split, idxs, float(alpha))
            if (
                single_tv_diagnostic_enabled
                and single_tv_val_best_acc is not None
                and single_tv_val_best_alpha is not None
            ):
                single_tv_by_idx = ev.eval_single_tv_indices(alpha_search_split, idxs, float(alpha))
                for idx in idxs:
                    if single_tv_by_idx[idx] > single_tv_val_best_acc[idx]:
                        single_tv_val_best_acc[idx] = float(single_tv_by_idx[idx])
                        single_tv_val_best_alpha[idx] = float(alpha)
            baseline_accs = [baseline_by_idx[i] for i in idxs]
            rebase_accs = [rebase_by_idx[i] for i in idxs]

            print(
                f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
            )
            print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
            for i, item in enumerate(per_task):
                task_name = item["task"]
                baseline_acc = baseline_accs[i]
                rebase_acc = rebase_accs[i]
                norm = _norm_acc(rebase_acc, baseline_acc)
                print(
                    f"  {task_name:<{task_col}}  {baseline_acc:>{metric_col}.6f}  {rebase_acc:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                )

            avg_rebase = average_scores(rebase_accs)
            avg_baseline = _average_defined(baseline_accs)
            avg_norm = _average_defined([_norm_acc(r, b) for r, b in zip(rebase_accs, baseline_accs, strict=True)])
            print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
            print(
                f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
            )

            sweep_results.append(
                {
                    "alpha": float(alpha),
                    "baseline_accs": baseline_accs,
                    "rebase_accs": rebase_accs,
                }
            )
            run_logger.log_event(
                "alpha_eval_end",
                metrics={
                    "alpha/value": float(alpha),
                    "alpha/avg_acc": float(avg_rebase),
                    "alpha/avg_norm_acc": float(avg_norm),
                },
                context={
                    "baseline_label": baseline_label,
                    "per_task_baseline": {item["task"]: float(baseline_accs[i]) for i, item in enumerate(per_task)},
                    "per_task_rebased": {item["task"]: float(rebase_accs[i]) for i, item in enumerate(per_task)},
                },
            )

            eps = 1e-12
            # Track baseline best alpha independently of rebased best alpha,
            # but only when there is at least one untransported baseline task
            # (otherwise the baseline is target_zeroshot and alpha-independent).
            if has_untransported:
                if avg_baseline != avg_baseline:  # NaN guard
                    avg_baseline_for_track = float("-inf")
                else:
                    avg_baseline_for_track = float(avg_baseline)
                if avg_baseline_for_track > best_baseline_avg + eps:
                    best_baseline_avg = avg_baseline_for_track
                    best_baseline_alpha = float(alpha)

            if avg_rebase > best_rebase_avg + eps:
                best_rebase_avg = avg_rebase
                best_alpha = float(alpha)
                shared_bad_steps = 0
            elif avg_rebase + eps >= best_rebase_avg:
                shared_bad_steps = 0
            elif len(sweep_alphas) > 1:
                shared_bad_steps += 1
                print(
                    f"  (alpha={alpha:.3f} fell below best shared avg {best_rebase_avg:.6f}; "
                    f"bad_steps={shared_bad_steps}/{alpha_patience + 1})"
                )
                if shared_bad_steps > alpha_patience:
                    break

        print("\n=== Alpha search summary (shared) ===")
        for r in sweep_results:
            a = r["alpha"]
            avg_r = average_scores(r["rebase_accs"])
            avg_b = _average_defined(r["baseline_accs"])
            print(f"  alpha={a:.3f}  {baseline_label}={avg_b:.6f}  {result_label}={avg_r:.6f}")
        global_alpha_curve = [
            {
                "alpha": float(row["alpha"]),
                "avg_rebased": float(average_scores(row["rebase_accs"])),
                "avg_baseline": float(_average_defined(row["baseline_accs"])),
                "per_task_rebased": {item["task"]: float(row["rebase_accs"][idx]) for idx, item in enumerate(per_task)},
                "per_task_baseline": {
                    item["task"]: float(row["baseline_accs"][idx]) for idx, item in enumerate(per_task)
                },
            }
            for row in sweep_results
        ]
        print(
            f"\nBest alpha: rebase={best_alpha:.3f} (avg rebased val acc={best_rebase_avg:.6f}) | "
            f"baseline={best_baseline_alpha:.3f} (avg baseline val acc={best_baseline_avg:.6f})"
        )

        print(
            f"\n(Re-running on test split: rebase at alpha={best_alpha:.3f}, baseline at alpha={best_baseline_alpha:.3f})"
        )
        all_indices = list(range(len(per_task)))
        baseline_test_by_idx = ev.eval_baseline_indices("test", all_indices, float(best_baseline_alpha))
        rebase_test_by_idx = ev.eval_rebased_indices("test", all_indices, float(best_alpha))
        baseline_test_accs = [baseline_test_by_idx[i] for i in all_indices]
        rebase_test_accs = [rebase_test_by_idx[i] for i in all_indices]
        selected_alpha_by_task = [float(best_alpha)] * len(per_task)
        selected_baseline_alpha_by_task = [float(best_baseline_alpha)] * len(per_task)

    else:
        tracker = PerTaskAlphaTracker(
            task_names=[str(item["task"]) for item in per_task],
            initial_alpha=float(alphas[0]),
            patience=alpha_patience,
        )
        # For tasks where the untransported baseline is infeasible (different
        # architecture / shape), the baseline is target_zeroshot and
        # alpha-independent. Pre-seed the secondary tracker so
        # best_secondary_alpha stays at 0.0 and the secondary stream never
        # participates in alpha optimization for these tasks.
        for idx in range(len(per_task)):
            if not can_eval_untransported_by_task[idx]:
                baseline_val = ev.eval_baseline(alpha_search_split, idx, 0.0)
                tracker.best_secondary_alpha[idx] = 0.0
                tracker.best_secondary_acc[idx] = baseline_val
                tracker.secondary_active[idx] = False
        sweep_results = []

        for alpha in alphas:
            # The baseline may keep being swept after the rebased has early-stopped
            # a task, so we evaluate the union of primary- and secondary-active tasks
            # and decouple the two streams' early stopping.
            eval_indices = tracker.eval_active_indices()
            if not eval_indices:
                print("\nAll tasks have early-stopped on both streams; ending per-task alpha sweep.")
                break

            primary_indices = set(tracker.primary_active_indices())
            print(f"\n=== alpha {alpha:.3f} — {method_label} (split: {alpha_search_split}, mode: per_task) ===")

            # Baseline must be evaluated for every index in the union (baseline may
            # still be active for tasks where the rebased already early-stopped).
            baseline_by_idx = ev.eval_baseline_indices(alpha_search_split, eval_indices, float(alpha))
            # Rebased is only evaluated for primary-active tasks; for tasks where it
            # has already stopped, we pass -inf as a no-op placeholder.
            rebase_eval_indices = [idx for idx in eval_indices if idx in primary_indices]
            rebase_by_idx_active = ev.eval_rebased_indices(alpha_search_split, rebase_eval_indices, float(alpha))
            rebase_by_idx: dict[int, float] = {}
            for idx in eval_indices:
                if idx in primary_indices:
                    rebase_by_idx[idx] = rebase_by_idx_active[idx]
                else:
                    rebase_by_idx[idx] = float("-inf")

            baseline_accs = [baseline_by_idx[idx] for idx in eval_indices]
            rebase_accs = [rebase_by_idx[idx] for idx in eval_indices]

            print(
                f"  {'task':<{task_col}}  {baseline_label:>{metric_col}}  {result_label:>{metric_col}}  {'norm':>{metric_col}}"
            )
            print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
            for idx, baseline_acc, rebase_acc in zip(eval_indices, baseline_accs, rebase_accs, strict=True):
                task_name = per_task[idx]["task"]
                # During the val sweep, norm uses the baseline's own best-so-far
                # accuracy (not the same-alpha baseline), so the per-step ratio
                # already reflects the final normalization semantics.
                best_secondary = float(tracker.best_secondary_acc[idx])
                norm_baseline = best_secondary if best_secondary != float("-inf") else baseline_acc
                # For tasks whose rebased stream already early-stopped, rebase_acc
                # is -inf (not evaluated this step); display the frozen best rebased
                # accuracy instead so the reported number tracks the rebased peak.
                if idx in primary_indices:
                    display_rebase = rebase_acc
                else:
                    frozen = float(tracker.best_primary_acc[idx])
                    display_rebase = frozen if frozen != float("-inf") else 0.0
                norm = _norm_acc(display_rebase, norm_baseline)
                marker = " " if idx in primary_indices else "*"
                print(
                    f" {marker}{task_name:<{task_col - 1}}  {baseline_acc:>{metric_col}.6f}  {display_rebase:>{metric_col}.6f}  {norm:>{metric_col}.6f}"
                )

            # Average rebased across ALL tasks: use the current value for active tasks
            # and the frozen best for stopped tasks, so the avg does not
            # collapse to 0.0 once every rebased task has early-stopped.
            all_rebase_vals: list[float] = []
            for idx in range(len(per_task)):
                if idx in primary_indices:
                    all_rebase_vals.append(float(rebase_by_idx[idx]))
                else:
                    frozen = float(tracker.best_primary_acc[idx])
                    all_rebase_vals.append(frozen if frozen != float("-inf") else 0.0)
            avg_rebase = average_scores(all_rebase_vals)
            avg_baseline = _average_defined(baseline_accs)
            avg_norm = _average_defined(
                [
                    _norm_acc(
                        float(rebase_by_idx[idx])
                        if idx in primary_indices
                        else max(float(tracker.best_primary_acc[idx]), 0.0),
                        baseline_accs[i],
                    )
                    for i, idx in enumerate(eval_indices)
                ]
            )
            print(f"  {'-' * task_col}  {'-' * metric_col}  {'-' * metric_col}  {'-' * metric_col}")
            print(
                f"  {'avg':<{task_col}}  {avg_baseline:>{metric_col}.6f}  {avg_rebase:>{metric_col}.6f}  {avg_norm:>{metric_col}.6f}"
            )

            stopped_primary: list[int] = []
            stopped_secondary: list[int] = []
            stopped_primary, stopped_secondary = tracker.update(
                alpha=float(alpha),
                indices=eval_indices,
                primary_accs=rebase_accs,
                secondary_accs=baseline_accs,
            )
            if stopped_primary:
                stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_primary)
                print(f"  Early-stopping REBASED tasks at alpha={alpha:.3f}: {stopped_names}")
            if stopped_secondary:
                stopped_names = ", ".join(str(per_task[idx]["task"]) for idx in stopped_secondary)
                print(f"  Early-stopping BASELINE tasks at alpha={alpha:.3f}: {stopped_names}")

            run_logger.log_event(
                "alpha_eval_end",
                metrics={
                    "alpha/value": float(alpha),
                    "alpha/avg_acc": float(avg_rebase),
                    "alpha/avg_norm_acc": float(avg_norm),
                },
                context={
                    "active_tasks": [per_task[idx]["task"] for idx in eval_indices],
                    "per_task_baseline": {per_task[idx]["task"]: float(baseline_by_idx[idx]) for idx in eval_indices},
                    "per_task_rebased": {per_task[idx]["task"]: float(rebase_by_idx[idx]) for idx in eval_indices},
                    "stopped_primary": [int(idx) for idx in stopped_primary],
                    "stopped_secondary": [int(idx) for idx in stopped_secondary],
                },
            )

            sweep_results.append(
                {
                    "alpha": float(alpha),
                    "active_indices": list(eval_indices),
                    "baseline_accs": baseline_accs,
                    "rebase_accs": rebase_accs,
                    "primary_active": [int(idx) for idx in eval_indices if idx in primary_indices],
                }
            )

        print("\n=== Alpha search summary (per-task) ===")
        for idx, item in enumerate(per_task):
            print(
                f"  {item['task']}: rebase_alpha={tracker.best_primary_alpha[idx]:.3f}  "
                f"rebase_val={tracker.best_primary_acc[idx]:.6f} | "
                f"baseline_alpha={tracker.best_secondary_alpha[idx]:.3f}  "
                f"baseline_val={tracker.best_secondary_acc[idx]:.6f}"
            )
        print(f"\nAvg per-task best rebase val acc: {tracker.best_avg():.6f}")
        if alpha_search_split == "val":
            # Preserve the scores used for selection separately from test
            # metrics so campaign selection never needs a test fallback.
            selected_validation_results = {
                "split": "val",
                "per_task_rebased": {
                    item["task"]: float(tracker.best_primary_acc[i]) for i, item in enumerate(per_task)
                },
                "avg_rebased": float(tracker.best_avg()),
            }
        best_baseline_vals = [float(v) for v in tracker.best_secondary_acc if v != float("-inf")]
        if best_baseline_vals:
            print(f"Avg per-task best baseline val acc: {sum(best_baseline_vals) / len(best_baseline_vals):.6f}")

        print("\n(Re-running per-task best alphas on test split — decoupled per stream)")
        baseline_test_accs: list[float] = []
        rebase_test_accs: list[float] = []
        selected_alpha_by_task: list[float] = []
        selected_baseline_alpha_by_task: list[float] = []
        for idx, item in enumerate(per_task):
            rebase_alpha = float(tracker.best_primary_alpha[idx])
            baseline_alpha = float(tracker.best_secondary_alpha[idx])
            selected_alpha_by_task.append(rebase_alpha)
            selected_baseline_alpha_by_task.append(baseline_alpha)
            print(f"  {item['task']}: rebase_alpha={rebase_alpha:.3f}  baseline_alpha={baseline_alpha:.3f}")

            baseline_test_accs.append(ev.eval_baseline("test", idx, baseline_alpha))

            rebase_test_accs.append(ev.eval_rebased_indices("test", [idx], rebase_alpha)[idx])
        best_alpha = float(sum(selected_alpha_by_task) / max(1, len(selected_alpha_by_task)))
        best_baseline_alpha = float(sum(selected_baseline_alpha_by_task) / max(1, len(selected_baseline_alpha_by_task)))

    norm_accs = [_norm_acc(r, b) for r, b in zip(rebase_test_accs, baseline_test_accs, strict=True)]

    single_tv_test_accs: list[float] | None = None
    single_tv_test_alpha_by_task: list[float] | None = None
    if single_tv_diagnostic_enabled and single_tv_val_best_alpha is not None:
        single_tv_test_alpha_by_task = [float(a) for a in single_tv_val_best_alpha]
        single_tv_test_accs = []
        print("\nSingle transported task-vector test diagnostic:")
        for idx, item in enumerate(per_task):
            alpha = single_tv_test_alpha_by_task[idx]
            single_acc = ev.eval_single_tv_indices("test", [idx], alpha)[idx]
            single_tv_test_accs.append(float(single_acc))
            print(f"  {item['task']}: alpha={alpha:.3f}  single_tv_test={single_acc:.6f}")
        single_avg = sum(single_tv_test_accs) / len(single_tv_test_accs)
        merged_avg = sum(rebase_test_accs) / len(rebase_test_accs)
        print(
            f"  avg single_tv_test={single_avg:.6f}  merged_test={merged_avg:.6f} "
            f"merge_gap={single_avg - merged_avg:+.6f}"
        )
        run_logger.log_event(
            "single_tv_test_diagnostic_end",
            metrics={
                "single_tv/avg_test_accuracy": float(single_avg),
                "single_tv/merged_test_accuracy": float(merged_avg),
                "single_tv/merge_gap": float(single_avg - merged_avg),
            },
            context={
                "alpha_protocol": single_tv_alpha_protocol,
                "per_task_alpha": {
                    item["task"]: float(single_tv_test_alpha_by_task[i]) for i, item in enumerate(per_task)
                },
                "per_task_test_accuracy": {
                    item["task"]: float(single_tv_test_accs[i]) for i, item in enumerate(per_task)
                },
            },
        )

    return AlphaResult(
        baseline_label=baseline_label,
        result_label=result_label,
        hierarchical=hierarchical,
        best_alpha=best_alpha,
        best_baseline_alpha=best_baseline_alpha,
        selected_alpha_by_task=selected_alpha_by_task,
        selected_baseline_alpha_by_task=selected_baseline_alpha_by_task,
        rebase_test_accs=rebase_test_accs,
        baseline_test_accs=baseline_test_accs,
        norm_accs=norm_accs,
        per_task_premerge_alphas=per_task_premerge_alphas,
        hierarchical_premerge_alpha_curve=hierarchical_premerge_alpha_curve,
        global_alpha_curve=global_alpha_curve,
        selected_validation_results=selected_validation_results,
        single_tv_test_accs=single_tv_test_accs,
        single_tv_alpha_protocol=single_tv_alpha_protocol,
        single_tv_val_best_alpha=single_tv_val_best_alpha,
        single_tv_val_best_acc=single_tv_val_best_acc,
        rebased_deltas=ev.rebased_deltas,
    )
