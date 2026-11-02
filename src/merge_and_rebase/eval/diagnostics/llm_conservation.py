"""LLM conservation check (docs/.local/AGENT_SHARED_SPACE.md, LLM track): what a transported update costs the target's
general capabilities at the selected alpha.

For a finished ``eval.llm_rebase`` run (its summary JSON: ``transported_tv.path`` from ``save_transported_tvs_dir``,
``best_alpha``, the target from the resolved config) or an explicit ``--tv/--alpha/--target``, the target base and
``target + alpha * tau`` are scored with the repo's lm-eval wrapper (``eval/llm_rebase/harness.run``) on

- **ARC-Challenge**, test split (1172 questions), 0-shot, log-likelihood multiple choice: ``acc`` and ``acc_norm``
  (``acc_norm``, length-normalised, is the usual ARC number);
- **MMLU-500**, a FIXED subset of 500 MMLU test questions, 5-shot (lm-eval's ``mmlu_<subject>`` tasks: the first
  5 dev questions of the subject as examples), ``acc``. The subset is a simple random sample without replacement of
  the 14042 pooled test questions of the 57 subjects (seed 0; pool ordered by lm-eval task name, then by the
  question's index in the subject's test split), so each subject appears in proportion to its size in expectation
  and ``mmlu500_acc`` (questions pooled, = lm-eval's size-weighted MMLU aggregate on the subset) is an unbiased
  estimate of the full test accuracy (binomial standard error ~2.2 points at 50 %). The doc indices per subject are
  stored in ``llm_conservation_mmlu500.json`` next to this file and never recomputed at run time.

Reported: both accuracies, the difference in points (rebased - base), and the binomial standard error of each
accuracy (the difference is paired on the same questions, so its own error is smaller; per-question outcomes are not
returned by the harness wrapper).

``missing_bias: materialize``: Llama / Qwen ``down_proj`` has no bias but the transported vector carries one. Every
bias key of ``tau`` missing from the model is added as a zero bias BEFORE any evaluation, the base included, so the
base and the rebased model run the same kernels and ``alpha * tau = 0`` reproduces the base numbers exactly.

Budget (one 16 GB GPU, 1.5B model, bf16, batch 8): ARC-Challenge 0-shot is ~4.7k short requests, MMLU-500 5-shot
~500 contexts of ~0.5-1k tokens (single-token choices share one forward per question); a few minutes per model,
~10 minutes for a run whose base is not cached yet. The base numbers are cached per target (and settings) in
``llm_conservation_base_<hash>.json`` in the cache directory (default: next to the output), computed once.

    python -m merge_and_rebase.eval.diagnostics.llm_conservation RUN_SUMMARY.json [--out OUT.json]
    python -m merge_and_rebase.eval.diagnostics.llm_conservation --tv TV.pt --alpha A --target HF_ID --out OUT.json
        [--dtype bfloat16] [--device cuda] [--batch-size 8] [--arc-fewshot 0] [--mmlu-fewshot 5] [--cache-dir DIR]

Default output: ``<summary dir>/<summary stem>_conservation.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

ARC_TASK = "arc_challenge"
MMLU_SUBSET_PATH = Path(__file__).resolve().with_name("llm_conservation_mmlu500.json")
MMLU_SUBSET_SIZE = 500
MMLU_SUBSET_SEED = 0
ARC_TEST_SIZE = 1172  # allenai/ai2_arc ARC-Challenge test
DEFAULTS = {"arc_fewshot": 0, "mmlu_fewshot": 5, "batch_size": 8}


# ---------------------------------------------------------------- the fixed MMLU subset


def mmlu_subset(sizes: Mapping[str, int], *, n: int = MMLU_SUBSET_SIZE, seed: int = MMLU_SUBSET_SEED) -> dict[str, list[int]]:
    """``{lm-eval task: sorted doc indices}``: ``n`` questions drawn uniformly without replacement from the pooled test
    questions (tasks in sorted name order, each task's questions in test-split order), ``torch.randperm`` seeded."""
    tasks = sorted(sizes)
    total = sum(int(sizes[t]) for t in tasks)
    if not 0 < n <= total:
        raise ValueError(f"subset size {n} not in (0, {total}]")
    picks = torch.randperm(total, generator=torch.Generator().manual_seed(int(seed)))[:n].tolist()
    starts, offset = {}, 0
    for t in tasks:
        starts[t] = offset
        offset += int(sizes[t])
    out: dict[str, list[int]] = {}
    bounds = [(starts[t], starts[t] + int(sizes[t]), t) for t in tasks]
    for p in picks:
        lo, _, t = next(b for b in bounds if b[0] <= p < b[1])
        out.setdefault(t, []).append(p - lo)
    return {t: sorted(out[t]) for t in sorted(out)}


def load_mmlu_subset(path: Path = MMLU_SUBSET_PATH) -> dict[str, list[int]]:
    """The stored subset; checked against a regeneration from its stored subject sizes."""
    data = json.loads(Path(path).read_text())
    samples = {t: [int(i) for i in v] for t, v in data["samples"].items()}
    if samples != mmlu_subset(data["sizes"], n=int(data["n"]), seed=int(data["seed"])):
        raise ValueError(f"{path}: stored MMLU subset does not match its definition (sizes, n, seed)")
    return samples


# ---------------------------------------------------------------- model and update


def attach_zero_biases(model: torch.nn.Module, keys) -> list[str]:
    """Add a zero bias to every ``nn.Linear`` named by a ``<module>.bias`` key missing from ``model`` (the
    ``missing_bias: materialize`` convention); returns the keys added."""
    present = set(model.state_dict())
    added = []
    for key in sorted(k for k in keys if k.endswith(".bias") and k not in present):
        module = model.get_submodule(key[: -len(".bias")])
        if not isinstance(module, torch.nn.Linear) or module.bias is not None:
            raise KeyError(f"{key}: cannot materialize a bias on {type(module).__name__}")
        module.bias = torch.nn.Parameter(
            torch.zeros(module.out_features, dtype=module.weight.dtype, device=module.weight.device),
            requires_grad=False,
        )
        added.append(key)
    return added


class ScaledUpdate:
    """Loads ``base + alpha * tau`` into the ``tau`` keys of ``model`` in place (fp32 sum, cast to the model dtype)."""

    def __init__(self, model: torch.nn.Module, tau: Mapping[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        missing = sorted(set(tau) - set(sd))
        if missing:
            raise KeyError(f"tau keys not in the target ({len(missing)}): {missing[:3]}")
        for k, v in tau.items():
            if tuple(v.shape) != tuple(sd[k].shape):
                raise ValueError(f"{k}: tau {tuple(v.shape)} vs target {tuple(sd[k].shape)}")
        self.model = model
        self.tau = {k: v.detach().to("cpu", torch.float32) for k, v in tau.items()}
        self.base = {k: sd[k].detach().to("cpu", torch.float32).clone() for k in tau}

    @torch.no_grad()
    def load(self, alpha: float) -> None:
        sd = self.model.state_dict()
        for k, t in self.tau.items():
            sd[k].copy_((self.base[k] + float(alpha) * t).to(sd[k].dtype))


# ---------------------------------------------------------------- scoring


def score(harness: Callable[..., dict[str, float]], model, tokenizer, *, subset: Mapping[str, list[int]],
          settings: Mapping[str, Any], device: str) -> dict[str, Any]:
    """ARC-Challenge (full test) and MMLU-500 in one harness call; ``mmlu500_acc`` pools the 500 questions."""
    mmlu_tasks = sorted(subset)
    fewshot = {ARC_TASK: int(settings["arc_fewshot"]), **{t: int(settings["mmlu_fewshot"]) for t in mmlu_tasks}}
    raw = harness(tasks=[ARC_TASK, *mmlu_tasks], model=model, tokenizer=tokenizer, device=device,
                  num_fewshot=fewshot, batch_size=int(settings["batch_size"]), samples=dict(subset))
    n_mmlu = sum(len(v) for v in subset.values())
    per_subject = {t: float(raw[f"{t}_acc"]) for t in mmlu_tasks}
    return {
        "arc_challenge_acc": float(raw[f"{ARC_TASK}_acc"]),
        "arc_challenge_acc_norm": float(raw[f"{ARC_TASK}_acc_norm"]),
        "mmlu500_acc": sum(per_subject[t] * len(subset[t]) for t in mmlu_tasks) / n_mmlu,
        "mmlu500_acc_by_subject": per_subject,
    }


HEADLINE = ("arc_challenge_acc_norm", "arc_challenge_acc", "mmlu500_acc")


def compare(base: Mapping[str, Any], rebased: Mapping[str, Any], n: Mapping[str, int]) -> dict[str, Any]:
    """Points (x100): rebased - base per headline metric, and the binomial standard error of each accuracy."""
    count = {"arc_challenge_acc_norm": n["arc_challenge"], "arc_challenge_acc": n["arc_challenge"],
             "mmlu500_acc": n["mmlu500"]}

    def se(p: float, k: int) -> float:
        return 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / k) if k else float("nan")

    return {
        "delta_points": {m: 100.0 * (float(rebased[m]) - float(base[m])) for m in HEADLINE},
        "se_points_base": {m: se(float(base[m]), count[m]) for m in HEADLINE},
        "se_points_rebased": {m: se(float(rebased[m]), count[m]) for m in HEADLINE},
    }


def base_cache_key(target: str, dtype: str, settings: Mapping[str, Any], subset: Mapping[str, list[int]],
                   bias_keys: list[str]) -> dict[str, Any]:
    try:
        import lm_eval

        version = getattr(lm_eval, "__version__", None)
    except ImportError:
        version = None
    subset_sha = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()
    biases_sha = hashlib.sha256(json.dumps(sorted(bias_keys)).encode()).hexdigest()
    return {"target": target, "dtype": dtype, **{k: settings[k] for k in sorted(DEFAULTS)}, "mmlu_subset": subset_sha,
            "tau_bias_keys": biases_sha, "lm_eval": version}


def evaluate_conservation(
    model,
    tokenizer,
    tau: Mapping[str, torch.Tensor],
    alpha: float,
    *,
    target: str,
    dtype: str,
    device: str,
    settings: Mapping[str, Any] | None = None,
    cache_dir: Path,
    subset: Mapping[str, list[int]] | None = None,
    harness: Callable[..., dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Score the target base (cached) and ``target + alpha * tau`` and compare them (see the module docstring)."""
    if harness is None:
        from ..llm_rebase.harness import run as harness
    settings = {**DEFAULTS, **dict(settings or {})}
    subset = load_mmlu_subset() if subset is None else {t: list(v) for t, v in subset.items()}
    biases = attach_zero_biases(model, tau)
    update = ScaledUpdate(model, tau)
    # keyed on every bias key of tau (whether added now or by an earlier call on this model object): the zero-bias
    # layout of the base model, identical for every vector of one method family on one target
    key = base_cache_key(target, dtype, settings, subset, sorted(k for k in tau if k.endswith(".bias")))
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"llm_conservation_base_{digest}.json"
    started = time.perf_counter()
    if cache_path.exists():
        base = json.loads(cache_path.read_text())["scores"]
        base_cached = True
    else:
        update.load(0.0)
        base = score(harness, model, tokenizer, subset=subset, settings=settings, device=device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key, "scores": base}, indent=1) + "\n")
        base_cached = False
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    update.load(alpha)
    try:
        rebased = score(harness, model, tokenizer, subset=subset, settings=settings, device=device)
    finally:
        update.load(0.0)
    n = {"arc_challenge": ARC_TEST_SIZE, "mmlu500": sum(len(v) for v in subset.values())}
    return {
        "target": target,
        "dtype": dtype,
        "alpha": float(alpha),
        "settings": {**settings, "arc_split": "test", "mmlu_subset": str(MMLU_SUBSET_PATH.name),
                     "mmlu_subset_seed": MMLU_SUBSET_SEED},
        "n_questions": n,
        "zero_biases_added": len(biases),
        "base": base,
        "rebased": rebased,
        **compare(base, rebased, n),
        "base_cache": str(cache_path),
        "base_cached": base_cached,
        "seconds": {"base": base_seconds, "rebased": time.perf_counter() - started},
    }


# ---------------------------------------------------------------- CLI


def run_inputs(summary_path: Path) -> dict[str, Any]:
    """``tv``, ``alpha``, ``target``, ``dtype`` of an llm_rebase run summary (``transported_tv``, ``best_alpha``)."""
    summary = json.loads(Path(summary_path).read_text())
    tv = (summary.get("transported_tv") or {}).get("path")
    if not tv:
        raise ValueError(f"{summary_path}: no transported_tv.path (run with save_transported_tvs_dir set)")
    cfg = ((summary.get("run_logging") or {}).get("metadata") or {}).get("resolved_config") or {}
    target = cfg.get("target_model_name_or_path")
    sidecar = (summary.get("transported_tv") or {}).get("sidecar")
    if target is None and sidecar and Path(sidecar).exists():
        target = json.loads(Path(sidecar).read_text()).get("refs", {}).get("target_model_name_or_path")
    if target is None:
        raise ValueError(f"{summary_path}: cannot read the target model (resolved_config or sidecar refs)")
    return {"tv": tv, "alpha": float(summary["best_alpha"]), "target": target,
            "dtype": _DTYPE_ALIASES.get(str(cfg.get("dtype") or "bfloat16"), str(cfg.get("dtype") or "bfloat16")),
            "tv_sha256": summary["transported_tv"].get("sha256")}


_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}  # llm_rebase config spellings


def load_target(target: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=getattr(torch, dtype)).to(device).eval()
    return model, tok


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary", nargs="?", default=None)
    ap.add_argument("--tv", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--arc-fewshot", type=int, default=DEFAULTS["arc_fewshot"])
    ap.add_argument("--mmlu-fewshot", type=int, default=DEFAULTS["mmlu_fewshot"])
    args = ap.parse_args(argv)

    inputs: dict[str, Any] = run_inputs(Path(args.summary)) if args.summary else {"dtype": "bfloat16"}
    for name in ("tv", "alpha", "target", "dtype"):  # explicit flags override the summary
        if getattr(args, name) is not None:
            inputs[name] = getattr(args, name)
    missing = [k for k in ("tv", "alpha", "target") if inputs.get(k) is None]
    if missing:
        raise SystemExit(f"missing {missing}: give a run summary or --tv/--alpha/--target")
    if args.out:
        out_path = Path(args.out)
    elif args.summary:
        s = Path(args.summary).resolve()
        out_path = s.with_name(f"{s.stem}_conservation.json")
    else:
        raise SystemExit("--out is required without a run summary")
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_path.parent
    tau = torch.load(inputs["tv"], map_location="cpu", weights_only=True)
    model, tok = load_target(inputs["target"], inputs["dtype"], args.device)
    settings = {"arc_fewshot": args.arc_fewshot, "mmlu_fewshot": args.mmlu_fewshot, "batch_size": args.batch_size}
    result = evaluate_conservation(model, tok, tau, inputs["alpha"], target=inputs["target"], dtype=inputs["dtype"],
                                   device=args.device, settings=settings, cache_dir=cache_dir)
    result = {"summary": None if not args.summary else str(Path(args.summary).resolve()), "tv": str(inputs["tv"]),
              "tv_sha256": inputs.get("tv_sha256"), **result}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    d = result["delta_points"]
    print(f"[llm_conservation] alpha {result['alpha']:g} | ARC-C acc_norm {100 * result['rebased']['arc_challenge_acc_norm']:.2f} "
          f"({d['arc_challenge_acc_norm']:+.2f}) | MMLU-500 {100 * result['rebased']['mmlu500_acc']:.2f} "
          f"({d['mmlu500_acc']:+.2f}) -> {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
