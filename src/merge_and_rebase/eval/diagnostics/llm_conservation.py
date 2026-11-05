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

- **Perplexity on generic text** (``wikitext103_ppl``): token-level perplexity, exp(total NLL / total predicted
  tokens), on a FIXED slice of WikiText-103 raw: ``PPL_DOCS`` = 256 paragraphs of >= 600 characters of the TEST split
  (seeded ``torch.randperm``, seed 0, in dataset order), each truncated to ``PPL_MAX_LENGTH`` = 512 tokens (the
  tokenizer's own special tokens, BOS where it adds one; the first token of a document without BOS is not
  predicted). Disjointness from the anchors frame: an ``anchors`` LLM run draws its generic text from the
  VALIDATION split of the same dataset (``eval/llm_rebase/anchors_pool.POOL_DATASETS``); this slice comes from a
  different split, and any paragraph whose text also occurs in the validation split is dropped (count recorded as
  ``n_overlap_removed``), so no anchor text can be a perplexity document. Docs, max length, split, seed and the
  sha256 of the texts are recorded under ``ppl_corpus`` in the output. Reported: base and rebased perplexity and
  their ratio (rebased / base) and the difference of mean NLL per token (nats).

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
        [--dtype bfloat16] [--device cuda] [--batch-size 2] [--arc-fewshot 0] [--mmlu-fewshot 5] [--cache-dir DIR]

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
DEFAULTS = {"arc_fewshot": 0, "mmlu_fewshot": 5, "batch_size": 2}  # 8: CUDA OOM on 16 GB (5-shot MMLU, 151k Qwen vocab)
PPL_DOCS = 256
PPL_MAX_LENGTH = 512
PPL_MIN_CHARS = 600
PPL_SEED = 0
PPL_SPLIT = "test"  # the anchors pool reads "validation" (anchors_pool.POOL_DATASETS): a different split
PPL_DATASET = "wikitext103"


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


# ---------------------------------------------------------------- perplexity corpus


def load_ppl_texts(*, n_docs: int = PPL_DOCS, seed: int = PPL_SEED, min_chars: int = PPL_MIN_CHARS) -> tuple[list[str], dict[str, Any]]:
    """The fixed perplexity paragraphs and their record (see the module docstring): ``n_docs`` paragraphs of at least
    ``min_chars`` characters of the WikiText-103 TEST split, chosen by a seeded permutation, minus any paragraph that
    also occurs in the split the anchors pool draws from."""
    import datasets

    from ..llm_rebase.anchors_pool import POOL_DATASETS

    path, name, pool_split, column = POOL_DATASETS[PPL_DATASET]
    if PPL_SPLIT == pool_split:
        raise ValueError(f"perplexity split {PPL_SPLIT!r} equals the anchors pool split; they must differ")
    pool_texts = {t for t in datasets.load_dataset(path, name, split=pool_split)[column] if isinstance(t, str)}
    ds = datasets.load_dataset(path, name, split=PPL_SPLIT)
    paragraphs = [t for t in ds[column] if isinstance(t, str) and len(t) >= min_chars]
    kept = [t for t in paragraphs if t not in pool_texts]
    if len(kept) < n_docs:
        raise ValueError(f"perplexity corpus: {len(kept)} paragraphs of >= {min_chars} chars, need {n_docs}")
    order = torch.randperm(len(kept), generator=torch.Generator().manual_seed(int(seed)))
    texts = [kept[i] for i in sorted(order[:n_docs].tolist())]
    return texts, ppl_corpus_record(texts, n_overlap_removed=len(paragraphs) - len(kept), seed=seed, min_chars=min_chars)


def ppl_corpus_record(texts, *, n_overlap_removed: int | None = None, seed: int = PPL_SEED, min_chars: int = PPL_MIN_CHARS) -> dict[str, Any]:
    return {"dataset": PPL_DATASET, "split": PPL_SPLIT, "excluded_split": "validation (anchors pool)",
            "min_chars": min_chars, "seed": seed, "n_docs": len(texts), "max_length": PPL_MAX_LENGTH,
            "n_overlap_removed": n_overlap_removed,
            "texts_sha256": hashlib.sha256(json.dumps(list(texts)).encode()).hexdigest()}


@torch.no_grad()
def perplexity(model, tokenizer, texts, *, max_length: int = PPL_MAX_LENGTH, batch_size: int = DEFAULTS["batch_size"],
               device: str = "cpu") -> dict[str, float]:
    """Token-level perplexity: ``exp(sum NLL / n_tokens)`` over every predicted token of every text (each text truncated to
    ``max_length`` tokens, right-padded, pad positions masked). Returns ``ppl``, ``nll`` (mean per token, nats), ``tokens``."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for i in range(0, len(texts), int(batch_size)):
            rows = tokenizer(list(texts[i:i + int(batch_size)]), truncation=True, max_length=int(max_length))["input_ids"]
            width = max(len(r) for r in rows)  # manual right padding: independent of the tokenizer's pad token / side
            ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=device)
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                                    reduction="none").reshape(ids[:, 1:].shape)
            valid = mask[:, 1:].bool()
            total += float(nll.double()[valid].sum())
            count += int(valid.sum())
    finally:
        model.train(was_training)
    return {"nll": total / count, "ppl": math.exp(total / count), "tokens": count}


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
          settings: Mapping[str, Any], device: str, ppl_texts=None) -> dict[str, Any]:
    """ARC-Challenge (full test) and MMLU-500 in one harness call; ``mmlu500_acc`` pools the 500 questions. With
    ``ppl_texts`` also the token-level perplexity on them (``wikitext103_ppl``, ``wikitext103_nll``)."""
    mmlu_tasks = sorted(subset)
    fewshot = {ARC_TASK: int(settings["arc_fewshot"]), **{t: int(settings["mmlu_fewshot"]) for t in mmlu_tasks}}
    raw = harness(tasks=[ARC_TASK, *mmlu_tasks], model=model, tokenizer=tokenizer, device=device,
                  num_fewshot=fewshot, batch_size=int(settings["batch_size"]), samples=dict(subset))
    n_mmlu = sum(len(v) for v in subset.values())
    per_subject = {t: float(raw[f"{t}_acc"]) for t in mmlu_tasks}
    out = {
        "arc_challenge_acc": float(raw[f"{ARC_TASK}_acc"]),
        "arc_challenge_acc_norm": float(raw[f"{ARC_TASK}_acc_norm"]),
        "mmlu500_acc": sum(per_subject[t] * len(subset[t]) for t in mmlu_tasks) / n_mmlu,
        "mmlu500_acc_by_subject": per_subject,
    }
    if ppl_texts is not None:
        p = perplexity(model, tokenizer, ppl_texts, max_length=int(settings.get("ppl_max_length", PPL_MAX_LENGTH)),
                       batch_size=int(settings["batch_size"]), device=device)
        out.update({"wikitext103_ppl": p["ppl"], "wikitext103_nll": p["nll"], "wikitext103_tokens": p["tokens"]})
    return out


HEADLINE = ("arc_challenge_acc_norm", "arc_challenge_acc", "mmlu500_acc")


def compare(base: Mapping[str, Any], rebased: Mapping[str, Any], n: Mapping[str, int]) -> dict[str, Any]:
    """Points (x100): rebased - base per headline metric, and the binomial standard error of each accuracy."""
    count = {"arc_challenge_acc_norm": n["arc_challenge"], "arc_challenge_acc": n["arc_challenge"],
             "mmlu500_acc": n["mmlu500"]}

    def se(p: float, k: int) -> float:
        return 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / k) if k else float("nan")

    out = {
        "delta_points": {m: 100.0 * (float(rebased[m]) - float(base[m])) for m in HEADLINE},
        "se_points_base": {m: se(float(base[m]), count[m]) for m in HEADLINE},
        "se_points_rebased": {m: se(float(rebased[m]), count[m]) for m in HEADLINE},
    }
    if "wikitext103_ppl" in base and "wikitext103_ppl" in rebased:
        out["ppl"] = {"base": float(base["wikitext103_ppl"]), "rebased": float(rebased["wikitext103_ppl"]),
                      "ratio_rebased_over_base": float(rebased["wikitext103_ppl"]) / float(base["wikitext103_ppl"]),
                      "delta_nll_nats": float(rebased["wikitext103_nll"]) - float(base["wikitext103_nll"])}
    return out


def base_cache_key(target: str, dtype: str, settings: Mapping[str, Any], subset: Mapping[str, list[int]],
                   bias_keys: list[str], ppl_texts=None) -> dict[str, Any]:
    try:
        import lm_eval

        version = getattr(lm_eval, "__version__", None)
    except ImportError:
        version = None
    subset_sha = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()
    biases_sha = hashlib.sha256(json.dumps(sorted(bias_keys)).encode()).hexdigest()
    key = {"target": target, "dtype": dtype, **{k: settings[k] for k in sorted(DEFAULTS)}, "mmlu_subset": subset_sha,
           "tau_bias_keys": biases_sha, "lm_eval": version}
    if ppl_texts is not None:  # absent without perplexity: the keys (and caches) of runs without it are unchanged
        key["ppl"] = {"texts_sha256": hashlib.sha256(json.dumps(list(ppl_texts)).encode()).hexdigest(),
                      "max_length": int(settings.get("ppl_max_length", PPL_MAX_LENGTH))}
    return key


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
    ppl_texts: Any = "default",
    ppl_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the target base (cached) and ``target + alpha * tau`` and compare them (see the module docstring).

    ``ppl_texts``: "default" loads the fixed WikiText-103 test slice (``load_ppl_texts``); a list of texts replaces
    it (tests); None skips perplexity."""
    if harness is None:
        from ..llm_rebase.harness import run as harness
    settings = {**DEFAULTS, **dict(settings or {})}
    if isinstance(ppl_texts, str) and ppl_texts == "default":
        ppl_texts, ppl_record = load_ppl_texts()
    elif ppl_texts is not None and ppl_record is None:
        ppl_record = ppl_corpus_record(ppl_texts, n_overlap_removed=None)
    subset = load_mmlu_subset() if subset is None else {t: list(v) for t, v in subset.items()}
    biases = attach_zero_biases(model, tau)
    update = ScaledUpdate(model, tau)
    # keyed on every bias key of tau (whether added now or by an earlier call on this model object): the zero-bias
    # layout of the base model, identical for every vector of one method family on one target
    key = base_cache_key(target, dtype, settings, subset, sorted(k for k in tau if k.endswith(".bias")), ppl_texts)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"llm_conservation_base_{digest}.json"
    started = time.perf_counter()
    if cache_path.exists():
        base = json.loads(cache_path.read_text())["scores"]
        base_cached = True
    else:
        update.load(0.0)
        base = score(harness, model, tokenizer, subset=subset, settings=settings, device=device, ppl_texts=ppl_texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"key": key, "scores": base}, indent=1) + "\n")
        base_cached = False
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    update.load(alpha)
    try:
        rebased = score(harness, model, tokenizer, subset=subset, settings=settings, device=device, ppl_texts=ppl_texts)
    finally:
        update.load(0.0)
    n = {"arc_challenge": ARC_TEST_SIZE, "mmlu500": sum(len(v) for v in subset.values())}
    return {
        "target": target,
        "dtype": dtype,
        "alpha": float(alpha),
        "settings": {**settings, "arc_split": "test", "mmlu_subset": str(MMLU_SUBSET_PATH.name),
                     "mmlu_subset_seed": MMLU_SUBSET_SEED,
                     **({"ppl_max_length": int(settings.get("ppl_max_length", PPL_MAX_LENGTH))} if ppl_texts is not None else {})},
        "ppl_corpus": dict(ppl_record) if ppl_texts is not None else None,
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
    ap.add_argument("--no-ppl", action="store_true", help="skip the WikiText-103 perplexity")
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
                                   device=args.device, settings=settings, cache_dir=cache_dir,
                                   ppl_texts=None if args.no_ppl else "default")
    result = {"summary": None if not args.summary else str(Path(args.summary).resolve()), "tv": str(inputs["tv"]),
              "tv_sha256": inputs.get("tv_sha256"), **result}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    d = result["delta_points"]
    print(f"[llm_conservation] alpha {result['alpha']:g} | ARC-C acc_norm {100 * result['rebased']['arc_challenge_acc_norm']:.2f} "
          f"({d['arc_challenge_acc_norm']:+.2f}) | MMLU-500 {100 * result['rebased']['mmlu500_acc']:.2f} "
          f"({d['mmlu500_acc']:+.2f})"
          + (f" | wikitext103 ppl {result['ppl']['rebased']:.3f} (x{result['ppl']['ratio_rebased_over_base']:.4f})"
             if "ppl" in result else "") + f" -> {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
