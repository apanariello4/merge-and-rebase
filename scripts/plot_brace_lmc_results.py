"""Plot completed BRACE/LMC summaries.

Usage: PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/plot_brace_lmc_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path("results/brace_lmc_corrected")
OUT = ROOT / "plots"


def load(name: str) -> dict:
    return json.loads((ROOT / f"{name}.json").read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single = {mode: load(f"eurosat_{mode}") for mode in ("independent", "shared")}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    before = single["independent"]["source_lmc"][0]["before_brace"]
    axes[0].plot(before["alphas"], before["loss"], "--", color="black", label="before BRACE (same for both)")
    for mode, summary in single.items():
        after = summary["source_lmc"][0]["after_brace"]
        axes[0].plot(after["alphas"], after["loss"], label=f"{mode}: after BRACE")
        axes[1].plot(after["alphas"], after["loss_barrier_curve"], label=mode)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="EuroSAT source loss", xlabel="Interpolation α", ylabel="Cross-entropy")
    axes[1].set(title="Corrected loss minus endpoint chord", xlabel="Interpolation α", ylabel="Loss chord gap")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("EuroSAT source LMC: base checkpoint (α=0) → fine-tuned checkpoint (α=1)")
    fig.savefig(OUT / "eurosat_lmc.png", dpi=180)
    plt.close(fig)

    cross = load("eurosat_gtsrb_shared")["cross_task_source_lmc"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(cross["alphas"], cross["average_loss"], label="average loss")
    axes[0].plot(cross["alphas"], cross["loss_chord_gap"], label="loss chord gap")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Shared EuroSAT↔GTSRB LMC", xlabel="EuroSAT → GTSRB α", ylabel="Loss")
    for task, values in cross["per_task_accuracy"].items():
        axes[1].plot(cross["alphas"], values, label=task)
    axes[1].set(title="Per-task source accuracy along path", xlabel="EuroSAT → GTSRB α", ylabel="Accuracy")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Shared cross-task source LMC: EuroSAT fine-tuned (α=0) → GTSRB fine-tuned (α=1)")
    fig.savefig(OUT / "shared_cross_task_lmc.png", dpi=180)
    plt.close(fig)

    for pair, tasks, title in (
        ("eurosat_gtsrb", ("EuroSAT", "GTSRB"), "EuroSAT fine-tuned (α=0) → GTSRB fine-tuned (α=1)"),
        ("dtd_svhn", ("DTD", "SVHN"), "DTD fine-tuned (α=0) → SVHN fine-tuned (α=1)"),
    ):
        cross_by_mode = {
            mode: load(f"{pair}_{mode}")["cross_task_source_lmc"][0]
            for mode in ("shared", "independent")
        }
        fig, axes = plt.subplots(4, 2, figsize=(10, 11.0), constrained_layout=True)
        for mode, values in cross_by_mode.items():
            axes[0, 0].plot(values["alphas"], values["average_loss"], "-o", markersize=2.5, label=mode)
            axes[0, 1].plot(values["alphas"], values["loss_chord_gap"], "-o", markersize=2.5, label=mode)
            for task, losses in values["per_task_loss"].items():
                chord = [(1 - alpha) * losses[0] + alpha * losses[-1] for alpha in values["alphas"]]
                axes[1, tasks.index(task)].plot(
                    values["alphas"], [loss - ref for loss, ref in zip(losses, chord, strict=True)], "-o", markersize=2.5, label=mode
                )
            for task, accuracy in values["per_task_accuracy"].items():
                axes[2, tasks.index(task)].plot(values["alphas"], accuracy, "-o", markersize=2.5, label=mode)
            axes[3, 0].plot(values["alphas"], values["average_accuracy"], "-o", markersize=2.5, label=mode)
        axes[0, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 0].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[0, 0].set(title="Average source loss", xlabel=f"{tasks[0]} → {tasks[1]} α", ylabel="Cross-entropy")
        axes[0, 1].set(title="Loss minus endpoint chord", xlabel=f"{tasks[0]} → {tasks[1]} α", ylabel="Loss chord gap")
        for index, task in enumerate(tasks):
            axes[1, index].set(title=f"{task} loss minus endpoint chord", xlabel=f"{tasks[0]} → {tasks[1]} α", ylabel="Loss chord gap")
            axes[2, index].set(title=f"{task} source accuracy", xlabel=f"{tasks[0]} → {tasks[1]} α", ylabel="Accuracy")
        axes[3, 0].set(title=f"Mean source accuracy across {tasks[0]} and {tasks[1]}", xlabel=f"{tasks[0]} → {tasks[1]} α", ylabel="Accuracy")
        axes[3, 1].set_visible(False)
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0]):
            axis.legend(fontsize=8)
        fig.suptitle(f"Cross-task source LMC: corrected {title}")
        fig.savefig(OUT / ("cross_task_lmc_comparison.png" if pair == "eurosat_gtsrb" else f"{pair}_cross_task_lmc_comparison.png"), dpi=180)
        plt.close(fig)

    barrier_rows = []
    for pair, tasks in (("eurosat_gtsrb", ("EuroSAT", "GTSRB")), ("dtd_svhn", ("DTD", "SVHN"))):
        for mode in ("shared", "independent"):
            values = load(f"{pair}_{mode}")["cross_task_source_lmc"][0]
            barriers = []
            for task in tasks:
                loss = values["per_task_loss"][task]
                chord = [(1 - alpha) * loss[0] + alpha * loss[-1] for alpha in values["alphas"]]
                barriers.append(max(0.0, max(value - reference for value, reference in zip(loss, chord, strict=True))))
            barrier_rows.append((f"{pair.replace('_', '+')}\n{mode}", max(barriers)))
    residual = json.loads((ROOT / "residual_metrics_dtd_svhn.json").read_text())
    residual_rows = [("DTD", residual["per_task"]["DTD"]), ("SVHN", residual["per_task"]["SVHN"]), ("DTD+SVHN\nmerge", residual["equal_weight_task_arithmetic_merge"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    labels, barriers = zip(*barrier_rows, strict=True)
    bars = axes[0].bar(labels, barriers, color=["C0", "C1", "C0", "C1"])
    axes[0].set(title=r"Worst-task source barrier $B_{\max}=\max_k B_k$", ylabel="Positive loss chord gap")
    axes[0].bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    x = list(range(len(residual_rows)))
    width = 0.25
    for offset, key, label in ((-width, "shared_norm", "shared delta"), (0, "independent_norm", "independent delta"), (width, "residual_norm", "residual")):
        axes[1].bar([value + offset for value in x], [row[key] for _, row in residual_rows], width=width, label=label)
    axes[1].set(xticks=x, xticklabels=[label for label, _ in residual_rows], yscale="log", title="Transported target-vector norms", ylabel=r"$\ell_2$ norm (log scale)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Cross-task LMC barrier and shared-versus-independent coordinate mismatch")
    fig.savefig(OUT / "cross_task_barrier_and_residuals.png", dpi=180)
    plt.close(fig)

    all_task = {}
    for mode in ("shared", "independent"):
        path = ROOT / f"vision8_all_task_lmc_{mode}.json"
        if path.exists():
            all_task[mode] = json.loads(path.read_text())["all_task_source_lmc"][0]
    if all_task:
        tasks = next(iter(all_task.values()))["tasks"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        x = list(range(len(tasks)))
        width = 0.7 / len(all_task)
        for index, (mode, values) in enumerate(all_task.items()):
            offset = (index - (len(all_task) - 1) / 2) * width
            axes[0].bar(
                [value + offset for value in x],
                [values["per_task_max_loss_barrier"][task] for task in tasks],
                width=width,
                label=mode,
            )
        axes[0].set(xticks=x, xticklabels=tasks, title=r"All-task star-simplex $B_k$", ylabel="Positive loss chord gap")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].legend(fontsize=8)
        labels = ["joint average", "worst task"]
        for index, (mode, values) in enumerate(all_task.items()):
            offset = (index - (len(all_task) - 1) / 2) * width
            bars = axes[1].bar(
                [value + offset for value in range(2)],
                [values["max_joint_loss_barrier"], values["max_per_task_loss_barrier"]],
                width=width,
                label=mode,
            )
            axes[1].bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        axes[1].set(xticks=range(2), xticklabels=labels, title=r"$B_{\mathrm{joint}}$ versus $B_{\mathrm{worst}}$", ylabel="Positive loss chord gap")
        axes[1].legend(fontsize=8)
        fig.suptitle("Vision8 all-task source LMC: endpoint-to-uniform-barycenter rays")
        fig.savefig(OUT / "vision8_all_task_simplex_barrier.png", dpi=180)
        plt.close(fig)

    functional = {
        "raw": json.loads((ROOT / "functional_overlap_dtd_svhn.json").read_text()),
        "unit parameter norm": json.loads((ROOT / "functional_overlap_dtd_svhn_unit_norm.json").read_text()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    x = list(range(len(functional)))
    width = 0.35
    for offset, mode in ((-width / 2, "shared"), (width / 2, "independent")):
        pooled = [values["strategies"][mode]["pooled"] for values in functional.values()]
        axes[0].bar([value + offset for value in x], [row["feature_delta_cosine"] for row in pooled], width=width, label=mode)
        axes[1].bar([value + offset for value in x], [row["relative_nonadditivity"] for row in pooled], width=width, label=mode)
    for axis, title, ylabel in (
        (axes[0], "Feature-change cosine", "cosine"),
        (axes[1], "Feature non-additivity", "interaction / additive norm"),
    ):
        axis.set(xticks=x, xticklabels=list(functional), title=title, ylabel=ylabel)
        axis.legend(fontsize=8)
    fig.suptitle("DTD--SVHN pooled target-feature probe")
    fig.savefig(OUT / "functional_overlap_dtd_svhn.png", dpi=180)
    plt.close(fig)

    lambda_runs = (
        ("λ=0 independent", "gtsrb_independent_lambda0"),
        ("λ=0 shared", "gtsrb_shared_lambda0"),
        ("λ=1000 independent", "gtsrb_independent_lambda1000"),
        ("λ=1000 shared", "gtsrb_shared_lambda1000"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for label, name in lambda_runs:
        lmc = json.loads((ROOT / "lambda_sweep" / f"{name}.json").read_text())["source_lmc"][0]["after_brace"]
        axes[0].plot(lmc["alphas"], lmc["loss"], "-o", markersize=2.5, label=label)
        axes[1].plot(lmc["alphas"], lmc["loss_barrier_curve"], "-o", markersize=2.5, label=label)
    axes[1].axhline(0, color="black", linewidth=0.8)
    for axis in axes:
        axis.xaxis.set_minor_locator(MultipleLocator(0.05))
        axis.grid(axis="x", which="minor", alpha=0.2)
    axes[0].set(title="Corrected GTSRB source loss", xlabel="Base → fine-tuned α", ylabel="Cross-entropy")
    axes[1].set(title="Corrected loss minus endpoint chord", xlabel="Base → fine-tuned α", ylabel="Loss chord gap")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("GTSRB source LMC after BRACE correction: completed lambda ablations")
    fig.savefig(OUT / "gtsrb_lambda_lmc.png", dpi=180)
    plt.close(fig)

    rows = [("EuroSAT\nindependent", single["independent"]["test_results"]), ("EuroSAT\nshared", single["shared"]["test_results"])]
    shared_cross = load("eurosat_gtsrb_shared")["test_results"]
    rows += [
        ("EuroSAT\nshared cross", {"per_task_rebased": {"x": shared_cross["per_task_rebased"]["EuroSAT"]}, "per_task_baseline": {"x": shared_cross["per_task_baseline"]["EuroSAT"]}}),
        ("GTSRB\nshared cross", {"per_task_rebased": {"x": shared_cross["per_task_rebased"]["GTSRB"]}, "per_task_baseline": {"x": shared_cross["per_task_baseline"]["GTSRB"]}}),
    ]
    for name, label in (
        ("gtsrb_shared_lambda0", "GTSRB\nλ0 shared"),
        ("gtsrb_independent_lambda0", "GTSRB\nλ0 independent"),
        ("gtsrb_shared_lambda1000", "GTSRB\nλ1000 shared"),
        ("gtsrb_independent_lambda1000", "GTSRB\nλ1000 independent"),
    ):
        result = json.loads((ROOT / "lambda_sweep" / f"{name}.json").read_text())["test_results"]
        rows.append((label, result))
    labels = [label for label, _ in rows]
    rebased = [next(iter(item["per_task_rebased"].values())) for _, item in rows]
    baseline = [next(iter(item["per_task_baseline"].values())) for _, item in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.bar([v - 0.2 for v in x], baseline, width=0.4, label="target zero-shot")
    ax.bar([v + 0.2 for v in x], rebased, width=0.4, label="rebased")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 0.8), ylabel="Test accuracy")
    ax.legend()
    fig.suptitle("Final target-model test accuracy after validation-selected rebase alpha")
    fig.savefig(OUT / "completed_test_accuracy.png", dpi=180)


if __name__ == "__main__":
    main()
