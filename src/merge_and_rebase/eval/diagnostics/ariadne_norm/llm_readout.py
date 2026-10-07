"""Readout diagnostic (LLM): does an Ariadne task update move the target the way the NATIVE fine-tuned target moves?

``python -m merge_and_rebase.eval.diagnostics.ariadne_norm.llm_readout --config <json> --stage {generate,measure}
--device cuda``

Setting (one config): source base / source fine-tuned (reference run), target base (biases materialized as in the run),
and the NATIVE fine-tuned target (``native_model_name_or_path``, loaded like ``TextLM.build`` with the run's dtype).
Same tokenizer family: token ids are asserted identical on the held-out texts.

Stage ``generate``: the held-out harness docs (the docs the reference run scored; verified in count and against the
summary's ``calibration_provenance.holdout_sha256``) are rendered by the task's ``doc_to_text`` and fed to the native
model EXACTLY as the harness fed the reference run: raw prompt, no chat template (``simple_evaluate`` is called without
``apply_chat_template``), ``add_special_tokens=False`` (``HFLM.add_bos_token=False``). Greedy decoding (no sampling, no
repetition penalty), at most ``max_new_tokens`` (256), stop at the native model's eos ids. Result:
``<out_dir>/reference_responses.jsonl`` (+ ``.sha256``, ``generate.json``), write-once.

Stage ``measure``: all forward passes are teacher-forced on ``prompt + native response`` and are scored on RESPONSE rows
only: the row at sequence position ``t`` with ``P-1 <= t <= T-2`` (``P`` prompt tokens, ``T`` total) predicts response
token ``t+1``. For the geometry (M1) two more scopes are reported: ``first`` (position 0) and ``last_response``
(position ``T-1``, the last response token as input). Docs are split deterministically (``split_seed``) into a
selection and a report half; every strength / scale is chosen on the selection half (minimum mean KL) and reported
on the report half and on all docs. Metrics of a variant (token-weighted means over response rows): KL(native ||
variant), top-1 agreement with the native model, negative log-likelihood of the native response tokens.

- M1 per-layer geometry, target layer j: ``D_j`` = transported target as the fit builds it (source base / fine-tuned
  outputs of layer ``pairing[j]``, ``_aligned``, ``@ Q_j`` of the fit's own ``prepare_direct_residual_streaming``),
  ``N_j`` = native layer-j output minus target-base layer-j output; cos / norm ratio / projection coefficient of the
  mounted update's layer-j displacement against both (alpha in {1, best alpha}).
- M2 last layer (before the final norm; the model's own final norm + lm_head is the head): base, full mounted update,
  its aligned / orthogonal parts w.r.t. ``d = D_last`` and w.r.t. ``n = N_last``, target pushed in (``h + a d``), native
  pushed in (``h + b n``), and the vocabulary target (ridge least squares ``delta_h`` through the final RMSNorm
  Jacobian and the unembedding restricted to the union of the top-k tokens of source ft / source base / target base).
- M3 single-layer injection: target base with layer j output ``+ beta D_j`` (and ``+ beta N_j``) at every position.
- M4 proxy validity: metrics of the original update at several alphas and of each variant at its best alpha against
  the IFEval instruction-level strict accuracy of the reference summaries (Spearman).

Config keys: ``reference_summary``, ``reference_config``, ``alpha_star_summary``, ``reference_merged_state`` (see
``llm.py``), ``original_tv_path``, ``variant_runs`` ({name: summary json}; unit vector = (merged_state.pt next to the
summary - target base) / best_alpha), ``native_model_name_or_path`` (``Qwen/Qwen2.5-1.5B-Instruct``), ``out_dir``,
``max_new_tokens`` (256), ``batch_size`` (4), ``split_seed`` (0), ``expected_n_docs`` (141), ``alpha_grid`` (0..2 step
0.1), ``vocab_beta_grid``, ``inject_betas`` (0.25, 0.5, 1, 1.5), ``vocab_top_k`` (64), ``vocab_ridge_rel`` (1e-3),
``gate_head_tol`` (1e-3), ``gate_batches`` (2), ``head_chunk`` (256), ``vocab_chunk`` (64), ``m4_alphas``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import socket
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from merge_and_rebase.rebase.methods._ariadne.fit import _task_vector_sha256
from merge_and_rebase.rebase.methods._ariadne.layouts import _aligned
from merge_and_rebase.rebase.methods._ariadne.streaming import prepare_direct_residual_streaming
from merge_and_rebase.run_logging import _code_fingerprint

from .runner import _state_sha

Tensor = torch.Tensor

DEFAULT_ALPHA_GRID = [round(0.1 * k, 10) for k in range(21)]
DEFAULT_VOCAB_BETA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DEFAULT_M4_ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
INJECT_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
INJECT_BETAS = (0.25, 0.5, 1.0, 1.5)
SCOPES = ("response", "first", "last_response")
ALIGNED_FAMILIES = ("aligned", "orthogonal", "aligned_native", "orthogonal_native")
RESPONSES_NAME = "reference_responses.jsonl"


class GateError(RuntimeError):
    """A correctness gate of the LLM readout failed."""


# ---- pure helpers -------------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holdout_sha256(eval_samples: Mapping[str, Sequence[int]]) -> str:
    """The formula of ``CalibrationTexts.provenance()["holdout_sha256"]``."""
    return sha256_bytes(json.dumps({k: list(v) for k, v in sorted(eval_samples.items())}, sort_keys=True).encode())


def split_indices(n: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic selection / report halves of ``range(n)`` (sorted indices; selection gets the extra doc)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()
    k = (n + 1) // 2
    return sorted(perm[:k]), sorted(perm[k:])


def project_on(delta: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row projection of ``delta`` on ``v`` in float64: ``(aligned, orthogonal)``; rows with ``v == 0`` keep all
    of ``delta`` in the orthogonal part (like ``readout.decompose``)."""
    delta64, v64 = delta.double(), v.double()
    v_sq = (v64 * v64).sum(-1)
    dot = (delta64 * v64).sum(-1)
    c = torch.where(v_sq > 0, dot / v_sq.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(dot))
    aligned = c[:, None] * v64
    return aligned, delta64 - aligned


def native_reference(native_logits: Tensor) -> tuple[Tensor, Tensor]:
    """``(log-probabilities, argmax)`` of the native model's rows (float32)."""
    logp = torch.log_softmax(native_logits.float(), dim=-1)
    return logp, logp.argmax(-1)


def token_metrics_from_reference(ref: tuple[Tensor, Tensor], variant_logits: Tensor, targets: Tensor) -> Tensor:
    """``[R, 3]`` float64 per row: KL(native || variant), top-1 agreement, NLL of the native token under the variant."""
    logp_n, top_n = ref
    logp_v = torch.log_softmax(variant_logits.float(), dim=-1)
    kl = (logp_n.exp() * (logp_n - logp_v)).sum(-1)
    agree = (logp_v.argmax(-1) == top_n).float()
    nll = -logp_v.gather(-1, targets[:, None]).squeeze(-1)
    return torch.stack([kl, agree, nll], dim=-1).double()


def token_metrics(native_logits: Tensor, variant_logits: Tensor, targets: Tensor) -> Tensor:
    return token_metrics_from_reference(native_reference(native_logits), variant_logits, targets)


class Bank:
    """Per-doc sums ``[n_docs, 4]`` (kl, agreement, nll, token count) of every measured variant."""

    def __init__(self, n_docs: int, device: Any) -> None:
        self.n_docs, self.device = int(n_docs), device
        self.sums: dict[tuple[str, str, float], Tensor] = {}

    def add(self, key: tuple[str, str, float], doc_of_row: Tensor, metrics: Tensor) -> None:
        if key not in self.sums:
            self.sums[key] = torch.zeros(self.n_docs, 4, dtype=torch.float64, device=self.device)
        vals = torch.cat([metrics, torch.ones_like(metrics[:, :1])], dim=1)
        self.sums[key].index_add_(0, doc_of_row, vals)

    def cpu(self) -> dict[tuple[str, str, float], Tensor]:
        return {k: v.cpu() for k, v in self.sums.items()}


def summarize_sums(arr: Tensor, idx: Sequence[int]) -> dict[str, float]:
    """Token-weighted means over the docs ``idx`` of a ``[n_docs, 4]`` sum array."""
    s = arr[list(idx)].sum(0)
    n = float(s[3])
    if n == 0:
        return {"kl": float("nan"), "top1": float("nan"), "nll": float("nan"), "n_tokens": 0}
    return {"kl": float(s[0]) / n, "top1": float(s[1]) / n, "nll": float(s[2]) / n, "n_tokens": int(n)}


def summarize_family(
    bank: Mapping[tuple[str, str, float], Tensor], family: str, update: str, sel: Sequence[int], rep: Sequence[int]
) -> dict[str, Any]:
    """Curve over the strength parameter, the parameter selected on ``sel`` (minimum KL, first on ties) and its scores."""
    params = sorted(k[2] for k in bank if k[0] == family and k[1] == update)
    n_all = range(next(iter(bank.values())).shape[0])
    curve = {
        f"{p:g}": {
            "selection": summarize_sums(bank[(family, update, p)], sel),
            "report": summarize_sums(bank[(family, update, p)], rep),
            "all": summarize_sums(bank[(family, update, p)], n_all),
        }
        for p in params
    }
    kls = [curve[f"{p:g}"]["selection"]["kl"] for p in params]
    best = min(range(len(params)), key=lambda i: (math.inf if math.isnan(kls[i]) else kls[i], i))
    chosen = f"{params[best]:g}"
    return {
        "curve": curve,
        "selected_param": params[best],
        "selected_by": "min mean KL on the selection half",
        "selected": {
            "selection": curve[chosen]["selection"],
            "report": curve[chosen]["report"],
            "all": curve[chosen]["all"],
        },
    }


def rmsnorm_jacobian_rows(a: Tensor, h: Tensor, w: Tensor, eps: float) -> Tensor:
    """``(W_K diag(w)) @ dRMSNorm/dh`` per row. ``a = W_K`` ``[R, K, D]``, ``h`` ``[R, D]``, ``w`` ``[D]``.

    ``y = w * h / r`` with ``r = sqrt(mean(h^2) + eps)``: ``dy_i/dh_k = w_i (delta_ik / r - h_i h_k / (D r^3))``.
    """
    d = h.shape[-1]
    r = (h.pow(2).mean(-1) + eps).sqrt()[:, None, None]
    aw = a * w
    ah = aw @ h[..., None]
    return aw / r - ah * h[:, None, :] / (d * r**3)


def ridge_dual(jac: Tensor, dl: Tensor, lam: Tensor) -> Tensor:
    """``J^T (J J^T + lam I)^-1 dl`` per row (dual form; ``J`` ``[R, K, D]``, ``dl`` ``[R, K]``, ``lam`` ``[R]``)."""
    k = jac.shape[1]
    gram = jac @ jac.transpose(-1, -2) + lam[:, None, None] * torch.eye(k, dtype=jac.dtype, device=jac.device)
    x = torch.linalg.solve(gram, dl[..., None])
    return (jac.transpose(-1, -2) @ x).squeeze(-1)


def ridge_primal(jac: Tensor, dl: Tensor, lam: Tensor) -> Tensor:
    """``(J^T J + lam I)^-1 J^T dl`` per row: the same ridge solution in the primal form (reference for tests)."""
    d = jac.shape[-1]
    gram = jac.transpose(-1, -2) @ jac + lam[:, None, None] * torch.eye(d, dtype=jac.dtype, device=jac.device)
    return torch.linalg.solve(gram, (jac.transpose(-1, -2) @ dl[..., None]))[..., 0]


def vocab_target_delta_h(
    h_base: Tensor,
    ft_logits: Tensor,
    base_logits: Tensor,
    target_logits: Tensor,
    unembed: Tensor,
    norm_weight: Tensor,
    eps: float,
    *,
    top_k: int = 64,
    ridge_rel: float = 1e-3,
) -> tuple[Tensor, dict[str, float]]:
    """Hidden displacement that realises the source's logit change through the TARGET's head, per row.

    ``delta_logit = ft_logits - base_logits`` restricted to ``K`` = union of the top-``top_k`` tokens of the three logit
    rows, centred over ``K``; ``J = W_U[K] @ dRMSNorm/dh (h_base)``; ``delta_h = J^T (J J^T + lam I)^-1 delta_logit`` with
    ``lam = ridge_rel * trace(J J^T) / |K|``. Float64 throughout; returns float32 ``[R, D]`` and scalar diagnostics.
    """
    k = min(int(top_k), ft_logits.shape[-1])
    idx = torch.cat([x.float().topk(k, dim=-1).indices for x in (ft_logits, base_logits, target_logits)], dim=-1)
    idx = idx.sort(dim=-1).values
    valid = torch.ones_like(idx, dtype=torch.bool)
    valid[:, 1:] = idx[:, 1:] != idx[:, :-1]
    n_valid = valid.sum(-1)
    dl = (ft_logits.float().gather(-1, idx) - base_logits.float().gather(-1, idx)).double() * valid
    dl = (dl - (dl.sum(-1) / n_valid)[:, None]) * valid
    jac = rmsnorm_jacobian_rows(unembed[idx].double(), h_base.double(), norm_weight.double(), float(eps))
    jac = jac * valid[..., None]
    lam = ridge_rel * (jac * jac).sum((-1, -2)) / n_valid
    delta_h = ridge_dual(jac, dl, lam)
    fit = (jac @ delta_h[..., None]).squeeze(-1)
    rel = ((fit - dl).norm(dim=-1) / dl.norm(dim=-1).clamp_min(1e-300)).mean()
    return delta_h.float(), {"mean_k": float(n_valid.float().mean()), "mean_relative_fit_error": float(rel)}


def ranks(x: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


class PairAcc:
    """Streamed scalar sums of a pair of row-vector streams ``a``, ``b``: cosines, norm ratio, projection of a on b."""

    def __init__(self, device: Any) -> None:
        self.s = torch.zeros(6, dtype=torch.float64, device=device)

    def update(self, a: Tensor, b: Tensor) -> None:
        if a.shape[0] == 0:
            return
        a, b = a.double(), b.double()
        dot, na, nb = (a * b).sum(-1), (a * a).sum(-1), (b * b).sum(-1)
        ok = (na > 0) & (nb > 0)
        cos = torch.where(ok, dot / (na * nb).clamp_min(1e-300).sqrt(), torch.zeros_like(dot))
        self.s += torch.stack(
            [
                torch.tensor(float(a.shape[0]), dtype=torch.float64, device=a.device),
                dot.sum(),
                na.sum(),
                nb.sum(),
                cos.sum(),
                ok.sum().double(),
            ]
        )

    def result(self) -> dict[str, float]:
        n, dot, na, nb, cos_sum, cos_n = self.s.tolist()
        nan = float("nan")
        return {
            "n_rows": int(n),
            "cos_global": dot / math.sqrt(na * nb) if na > 0 and nb > 0 else nan,
            "cos_mean_row": cos_sum / cos_n if cos_n > 0 else nan,
            "norm_ratio_a_over_b": math.sqrt(na / nb) if nb > 0 else nan,
            "coef_of_a_on_b": dot / nb if nb > 0 else nan,
        }


def quantiles(values: Sequence[Tensor]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    v = torch.cat([x.reshape(-1).double().cpu() for x in values])
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {"n": 0}
    q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {"n": int(v.numel()), "mean": float(v.mean()), "q10": float(q[0]), "median": float(q[1]), "q90": float(q[2])}


# ---- model plumbing -----------------------------------------------------------------------------------------------


class Head:
    """The model's own final norm + lm_head, applied to a last-layer (pre-norm) hidden state in the model's dtype."""

    def __init__(self, model: torch.nn.Module, fam: Any) -> None:
        self.norm = fam.final_norm(model)
        self.lm = model.get_output_embeddings()
        self.dtype = self.lm.weight.dtype
        self.eps = float(self.norm.variance_epsilon)

    def __call__(self, h: Tensor) -> Tensor:
        return self.lm(self.norm(h.to(self.dtype)))


@torch.no_grad()
def run_model(
    model: torch.nn.Module,
    fam: Any,
    ids: Tensor,
    mask: Tensor,
    capture: Sequence[int] = (),
    replace: Mapping[int, Tensor] | None = None,
    logits: bool = False,
) -> tuple[dict[int, Tensor], Tensor | None]:
    """One forward over ``ids``; returns ``({layer: output [B, T, D] in the model dtype}, logits or None)``.

    ``replace[j]`` is ADDED to layer j's output before it flows on (and before it is captured). Hooks are removed on
    exit; the model is not modified.
    """
    blocks = fam.layers(model)
    store: dict[int, Tensor] = {}
    handles = []

    def add_hook(j: int, delta: Tensor):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            new = t + delta.to(t.dtype)
            return (new, *out[1:]) if isinstance(out, tuple) else new

        return blocks[j].register_forward_hook(hook)

    def cap_hook(j: int):
        def hook(_m, _i, out):
            store[j] = (out[0] if isinstance(out, tuple) else out).detach()

        return blocks[j].register_forward_hook(hook)

    try:
        for j, delta in (replace or {}).items():
            handles.append(add_hook(int(j), delta))
        for j in capture:
            handles.append(cap_hook(int(j)))
        if logits:
            out = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        else:
            fam.calibration_forward(model, {"input_ids": ids, "attention_mask": mask}, ids.device)
            out = None
    finally:
        for h in handles:
            h.remove()
    return store, out


def mount_update(model: torch.nn.Module, base: Mapping[str, Tensor], tv: Mapping[str, Tensor], alpha: float) -> None:
    """``param <- cast(base + alpha * tv)`` for the keys of ``tv`` (all float32 on the model's device)."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, v in tv.items():
            p = params[key]
            p.copy_((base[key] + float(alpha) * v).to(p.dtype))


def build_rows(prompt_lens: Sequence[int], total_lens: Sequence[int]) -> dict[str, Tensor]:
    """Row set of a batch: positions ``0``, ``P-1 .. T-2`` (response scope) and ``T-1`` of every sequence."""
    b, t, resp, first, last = [], [], [], [], []
    for i, (p, tot) in enumerate(zip(prompt_lens, total_lens, strict=True)):
        for pos in sorted({0, tot - 1, *range(p - 1, tot - 1)}):
            b.append(i)
            t.append(pos)
            resp.append(p - 1 <= pos <= tot - 2)
            first.append(pos == 0)
            last.append(pos == tot - 1)
    return {
        "b": torch.tensor(b),
        "t": torch.tensor(t),
        "response": torch.tensor(resp),
        "first": torch.tensor(first),
        "last_response": torch.tensor(last),
    }


def collate(docs: Sequence[Mapping[str, Any]], pad_id: int, device: Any) -> dict[str, Any]:
    seqs = [list(d["prompt_ids"]) + list(d["response_ids"]) for d in docs]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return {
        "ids": ids.to(device),
        "mask": mask.to(device),
        "P": [len(d["prompt_ids"]) for d in docs],
        "T": [len(s) for s in seqs],
    }


# ---- inputs of the measurement ------------------------------------------------------------------------------------


@dataclass
class Update:
    name: str
    tv: dict[str, Tensor]  # unit-strength task vector, CPU float32, keys = target state-dict keys
    best_alpha: float
    sha256: str
    source: str = ""


@dataclass
class ReadoutSetup:
    device: Any
    fam: Any
    target: torch.nn.Module  # target base (biases materialized); never modified
    mounted: torch.nn.Module  # a copy of the target the updates are mounted on (modified, reset by every mount)
    source_base: torch.nn.Module
    source_ft: torch.nn.Module
    native: torch.nn.Module
    pairing: Sequence[int]
    q: Mapping[int, Tensor]  # fit's Procrustes map per target layer [D_source, D_target]
    target_base_state: Mapping[str, Tensor]
    updates: Sequence[Update]
    pad_id: int = 0
    extra_models: dict[str, torch.nn.Module] = field(default_factory=dict)


def model_hashes(setup: ReadoutSetup) -> dict[str, str]:
    models = {
        "target": setup.target,
        "source_base": setup.source_base,
        "source_ft": setup.source_ft,
        "native": setup.native,
        **setup.extra_models,
    }
    return {k: _state_sha(m) for k, m in models.items()}


def _chunks(n: int, size: int) -> Iterator[slice]:
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))


# ---- the measurement ----------------------------------------------------------------------------------------------


@torch.no_grad()
def run_measure(setup: ReadoutSetup, docs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """M1-M4 over ``docs`` (dicts with ``prompt_ids`` / ``response_ids`` / ``doc_index``). Returns the record."""
    t_start = time.perf_counter()
    dev, fam = setup.device, setup.fam
    n_docs = len(docs)
    sel, rep = split_indices(n_docs, int(cfg.get("split_seed", 0)))
    batch_size = int(cfg.get("batch_size", 4))
    head_chunk = int(cfg.get("head_chunk", 256))
    vocab_chunk = int(cfg.get("vocab_chunk", 64))
    gate_tol = float(cfg.get("gate_head_tol", 1e-3))
    gate_batches = int(cfg.get("gate_batches", 2))
    alpha_grid = [float(a) for a in cfg.get("alpha_grid", DEFAULT_ALPHA_GRID)]
    vocab_betas = [float(b) for b in cfg.get("vocab_beta_grid", DEFAULT_VOCAB_BETA_GRID)]
    inject_betas = [float(b) for b in cfg.get("inject_betas", INJECT_BETAS)]
    m4_alphas = [float(a) for a in cfg.get("m4_alphas", DEFAULT_M4_ALPHAS)]
    top_k = int(cfg.get("vocab_top_k", 64))
    ridge_rel = float(cfg.get("vocab_ridge_rel", 1e-3))

    target, mounted, native = setup.target, setup.mounted, setup.native
    n_layers = len(fam.layers(target))
    last = n_layers - 1
    pairing = list(setup.pairing)
    src_last = len(fam.layers(setup.source_base)) - 1
    src_layers = sorted({*pairing, src_last})
    inject_layers = sorted({round(f * last) for f in INJECT_FRACTIONS} | {last})
    heads = {
        "target": Head(target, fam),
        "native": Head(native, fam),
        "src_base": Head(setup.source_base, fam),
        "src_ft": Head(setup.source_ft, fam),
    }
    q = {int(j): v.to(dev, torch.float32) for j, v in setup.q.items()}

    if heads["src_base"].lm.weight.shape[0] != heads["target"].lm.weight.shape[0]:
        raise ValueError("source and target vocabularies differ")
    keys = sorted(setup.updates[0].tv)
    base_vals = {k: setup.target_base_state[k].to(dev, torch.float32) for k in keys}
    tvs = {u.name: {k: u.tv[k].to(dev, torch.float32) for k in keys} for u in setup.updates}
    best = {u.name: float(u.best_alpha) for u in setup.updates}

    def alphas_of(u: Update) -> list[float]:
        extra = [1.0, best[u.name], *(m4_alphas if u.name == "original" else [])]
        return sorted({float(a) for a in [*alpha_grid, *extra]})

    alphas = {u.name: alphas_of(u) for u in setup.updates}
    m1_cfgs = [(u.name, a) for u in setup.updates for a in sorted({1.0, best[u.name]})]
    mount_state: dict[str, Any] = {"cur": None}

    def mount(name: str, alpha: float) -> None:
        if mount_state["cur"] != (name, alpha):
            mount_update(mounted, base_vals, tvs[name], alpha)
            mount_state["cur"] = (name, alpha)

    bank = Bank(n_docs, dev)
    m1 = {
        (j, scope, name): PairAcc(dev)
        for j in range(n_layers)
        for scope in SCOPES
        for name in ["D|N", *[f"{u}@{a:g}|{pair}" for u, a in m1_cfgs for pair in ("dT|N", "dT|D")]]
    }
    geometry = {k: [] for k in ("cos_d_n", "cos_vocab_n", "cos_vocab_d", "norm_d_over_n", "norm_vocab_over_n")}
    vocab_info: list[dict[str, float]] = []
    gate_head: dict[str, float] = {}
    gate_beta0: dict[str, bool] = {}
    gate_mount0 = {"max_abs": 0.0}

    for bi, start in enumerate(range(0, n_docs, batch_size)):
        batch_docs = docs[start : start + batch_size]
        cb = collate(batch_docs, setup.pad_id, dev)
        ids, mask = cb["ids"], cb["mask"]
        rows = build_rows(cb["P"], cb["T"])
        rb, rt_ = rows["b"].to(dev), rows["t"].to(dev)
        scope_mask = {s: rows[s].to(dev) for s in SCOPES}
        resp = scope_mask["response"]
        rr = (rb[resp], rt_[resp])
        doc_of_row = torch.tensor([start + int(b) for b in rows["b"][rows["response"]]], device=dev)
        targets = ids[rr[0], rr[1] + 1]

        # -- base-side streams -------------------------------------------------------------------------------------
        base_h, base_logits = run_model(target, fam, ids, mask, range(n_layers), logits=bi < gate_batches)
        nat_h, nat_logits = run_model(native, fam, ids, mask, range(n_layers), logits=bi < gate_batches)
        sb_h, _ = run_model(setup.source_base, fam, ids, mask, src_layers)
        sf_h, _ = run_model(setup.source_ft, fam, ids, mask, src_layers)

        if bi < gate_batches:
            for name, _model, store, lg in (
                ("target", target, base_h, base_logits),
                ("native", native, nat_h, nat_logits),
            ):
                head = heads["target" if name == "target" else "native"]
                m = mask.bool()
                diff = (head(store[last]).float() - lg.float())[m].abs().max().item()
                gate_head[name] = max(gate_head.get(name, 0.0), diff)
            u0 = setup.updates[0]
            mount(u0.name, best[u0.name])
            mh, mlg = run_model(mounted, fam, ids, mask, [last], logits=True)
            diff = (heads["target"](mh[last]).float() - mlg.float())[mask.bool()].abs().max().item()
            gate_head["mounted"] = max(gate_head.get("mounted", 0.0), diff)
        del base_logits, nat_logits

        # -- native reference (log-probabilities of the native rows) ------------------------------------------------
        h_base_r = base_h[last][rr].float()
        nat_chunks = []
        for sl in _chunks(h_base_r.shape[0], head_chunk):
            nat_chunks.append((sl, native_reference(heads["native"](nat_h[last][rr[0][sl], rr[1][sl]]))))

        def score(h_rows: Tensor, nat_chunks=nat_chunks, targets=targets) -> Tensor:
            out = []
            for sl, ref in nat_chunks:
                out.append(token_metrics_from_reference(ref, heads["target"](h_rows[sl]), targets[sl]))
            return torch.cat(out, 0)

        # -- mounted forwards of the M1 configurations (all layers) --------------------------------------------------
        m1_h: dict[tuple[str, float], dict[int, Tensor]] = {}
        for name, a in m1_cfgs:
            mount(name, a)
            m1_h[(name, a)], _ = run_model(mounted, fam, ids, mask, range(n_layers))

        # -- M1 geometry; D_j / N_j kept for the injection layers ----------------------------------------------------
        keep: dict[int, tuple[Tensor, Tensor]] = {}
        d_last = n_last = None
        for j in range(n_layers):
            i = pairing[j]
            t_ref = base_h[j].float()
            fa = _aligned([sf_h[i].float()], [t_ref])[0]
            ba = _aligned([sb_h[i].float()], [t_ref])[0]
            d_full = (fa - ba) @ q[j]
            n_full = nat_h[j].float() - t_ref
            if j in inject_layers:
                keep[j] = (d_full, n_full)
            if j == last:
                d_last, n_last = d_full[rr], n_full[rr]
            for scope in SCOPES:
                sm = scope_mask[scope]
                if not bool(sm.any()):
                    continue
                gi = (rb[sm], rt_[sm])
                dj, nj = d_full[gi], n_full[gi]
                m1[(j, scope, "D|N")].update(dj, nj)
                for name, a in m1_cfgs:
                    dt = m1_h[(name, a)][j].float()[gi] - t_ref[gi]
                    m1[(j, scope, f"{name}@{a:g}|dT|N")].update(dt, nj)
                    m1[(j, scope, f"{name}@{a:g}|dT|D")].update(dt, dj)
        del m1_h

        # -- M3: single-layer injection into the target base ---------------------------------------------------------
        valid = mask[..., None].float()
        for j in inject_layers:
            for kind, vec in (("D", keep[j][0]), ("N", keep[j][1])):
                for beta in ((0.0,) if bi == 0 else ()) + tuple(inject_betas):
                    st, _ = run_model(target, fam, ids, mask, [last], replace={j: beta * vec * valid})
                    if beta == 0.0:
                        gate_beta0[f"{kind}@{j}"] = bool(torch.equal(st[last], base_h[last]))
                        continue
                    bank.add((f"inject_{kind}", f"layer{j}", beta), doc_of_row, score(st[last][rr].float()))
        del keep

        # -- M2 ingredients: vocabulary target and geometry of d, n -------------------------------------------------
        sbl, sfl = sb_h[src_last][rr], sf_h[src_last][rr]
        unembed = heads["target"].lm.weight
        delta_h_vocab = torch.empty_like(h_base_r)
        for sl in _chunks(h_base_r.shape[0], vocab_chunk):
            dh, info = vocab_target_delta_h(
                h_base_r[sl],
                heads["src_ft"](sfl[sl]),
                heads["src_base"](sbl[sl]),
                heads["target"](h_base_r[sl]),
                unembed,
                heads["target"].norm.weight,
                heads["target"].eps,
                top_k=top_k,
                ridge_rel=ridge_rel,
            )
            delta_h_vocab[sl] = dh
            vocab_info.append(info)

        def cosine(a: Tensor, b: Tensor) -> Tensor:
            return (a.double() * b.double()).sum(-1) / (a.double().norm(dim=-1) * b.double().norm(dim=-1)).clamp_min(
                1e-300
            )

        geometry["cos_d_n"].append(cosine(d_last, n_last))
        geometry["cos_vocab_n"].append(cosine(delta_h_vocab, n_last))
        geometry["cos_vocab_d"].append(cosine(delta_h_vocab, d_last))
        geometry["norm_d_over_n"].append(d_last.double().norm(dim=-1) / n_last.double().norm(dim=-1).clamp_min(1e-300))
        geometry["norm_vocab_over_n"].append(
            delta_h_vocab.double().norm(dim=-1) / n_last.double().norm(dim=-1).clamp_min(1e-300)
        )

        # -- M2 variants -------------------------------------------------------------------------------------------
        bank.add(("base", "", 0.0), doc_of_row, score(h_base_r))
        for a in alpha_grid:
            bank.add(("target", "", a), doc_of_row, score(h_base_r + a * d_last))
            bank.add(("native_push", "", a), doc_of_row, score(h_base_r + a * n_last))
        for b in vocab_betas:
            bank.add(("vocab", "", b), doc_of_row, score(h_base_r + b * delta_h_vocab))
        for u in setup.updates:
            for a in alphas[u.name]:
                mount(u.name, a)
                mh, _ = run_model(mounted, fam, ids, mask, [last])
                h_up = mh[last][rr].float()
                if a == 0.0:
                    gate_mount0["max_abs"] = max(gate_mount0["max_abs"], float((h_up - h_base_r).abs().max()))
                delta = h_up - h_base_r
                bank.add(("full", u.name, a), doc_of_row, score(h_up))
                for vname, vec in (("", d_last), ("_native", n_last)):
                    al, orth = project_on(delta, vec)
                    bank.add((f"aligned{vname}", u.name, a), doc_of_row, score(h_base_r + al.float()))
                    bank.add((f"orthogonal{vname}", u.name, a), doc_of_row, score(h_base_r + orth.float()))
        del base_h, nat_h, sb_h, sf_h, nat_chunks

    # -- aggregation -----------------------------------------------------------------------------------------------
    cpu_bank = bank.cpu()
    all_docs = list(range(n_docs))
    families: dict[str, Any] = {}
    for fam_name in ("base", "target", "native_push", "vocab"):
        families[fam_name] = summarize_family(cpu_bank, fam_name, "", sel, rep)
    inject: dict[str, Any] = {}
    for kind in ("D", "N"):
        for j in inject_layers:
            inject[f"{kind}@layer{j}"] = summarize_family(cpu_bank, f"inject_{kind}", f"layer{j}", sel, rep)
    updates_out: dict[str, Any] = {}
    for u in setup.updates:
        entry: dict[str, Any] = {"best_alpha": best[u.name], "sha256": u.sha256, "source": u.source}
        for fam_name in ("full", *ALIGNED_FAMILIES):
            s = summarize_family(cpu_bank, fam_name, u.name, sel, rep)
            for tag, a in (("at_best_alpha", best[u.name]), ("at_alpha_1", 1.0)):
                arr = cpu_bank[(fam_name, u.name, a)]
                s[tag] = {
                    "alpha": a,
                    "selection": summarize_sums(arr, sel),
                    "report": summarize_sums(arr, rep),
                    "all": summarize_sums(arr, all_docs),
                }
            entry[fam_name] = s
        updates_out[u.name] = entry

    m1_out: dict[str, Any] = {}
    for j in range(n_layers):
        m1_out[str(j)] = {
            "source_layer": int(pairing[j]),
            **{
                scope: {
                    name: m1[(j, scope, name)].result()
                    for name in ["D|N", *[f"{u}@{a:g}|{pair}" for u, a in m1_cfgs for pair in ("dT|N", "dT|D")]]
                }
                for scope in SCOPES
            },
        }

    m4_rows = []
    for u in setup.updates:
        for a in alphas[u.name]:
            if (u.name == "original" and any(abs(a - m) < 1e-9 for m in m4_alphas)) or abs(a - best[u.name]) < 1e-12:
                arr = cpu_bank[("full", u.name, a)]
                m4_rows.append(
                    {
                        "update": u.name,
                        "alpha": a,
                        "is_best_alpha": abs(a - best[u.name]) < 1e-12,
                        "selection": summarize_sums(arr, sel),
                        "report": summarize_sums(arr, rep),
                        "all": summarize_sums(arr, all_docs),
                    }
                )

    gates = {
        "head_matches_forward": {
            "max_abs_diff": gate_head,
            "tolerance": gate_tol,
            "passed": bool(all(v < gate_tol for v in gate_head.values())),
        },
        "injection_beta0_equals_base": {
            "by_layer": gate_beta0,
            "passed": bool(all(gate_beta0.values()) and gate_beta0),
        },
        "mount_alpha0_equals_base": {"max_abs_diff": gate_mount0["max_abs"], "passed": gate_mount0["max_abs"] == 0.0},
    }
    return {
        "n_docs": n_docs,
        "n_response_tokens": int(next(iter(cpu_bank.values()))[:, 3].sum()),
        "split": {"seed": int(cfg.get("split_seed", 0)), "selection": sel, "report": rep},
        "alpha_grid": alpha_grid,
        "vocab_beta_grid": vocab_betas,
        "inject_layers": inject_layers,
        "inject_betas": inject_betas,
        "m1": m1_out,
        "m2": {
            "variants": families,
            "updates": updates_out,
            "geometry_last_layer": {k: quantiles(v) for k, v in geometry.items()},
            "vocab_solver": {
                "mean_k": float(np.mean([v["mean_k"] for v in vocab_info])),
                "mean_relative_fit_error": float(np.mean([v["mean_relative_fit_error"] for v in vocab_info])),
                "top_k": top_k,
                "ridge_rel": ridge_rel,
            },
        },
        "m3": inject,
        "m4": {"rows": m4_rows},
        "gates": gates,
        "seconds": time.perf_counter() - t_start,
    }


def ifeval_inst_strict(results: Mapping[str, float] | None) -> float | None:
    if not results:
        return None
    for k, v in results.items():
        if "inst_level_strict_acc" in k and "stderr" not in k:
            return float(v)
    return None


def attach_m4_scores(rows: list[dict[str, Any]], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Adds the IFEval scores of ``summaries[update]["harness_results_by_alpha"]`` to the M4 rows; Spearman per proxy."""
    for row in rows:
        by_alpha = (summaries.get(row["update"]) or {}).get("harness_results_by_alpha") or {}
        res = by_alpha.get(f"{row['alpha']:g}")
        row["ifeval"] = dict(res) if res else None
        row["ifeval_inst_level_strict_acc"] = ifeval_inst_strict(res)
    usable = [r for r in rows if r["ifeval_inst_level_strict_acc"] is not None]
    out: dict[str, Any] = {"n_points": len(usable)}
    for split in ("report", "all"):
        out[split] = {
            m: spearman([r[split][m] for r in usable], [r["ifeval_inst_level_strict_acc"] for r in usable])
            for m in ("kl", "top1", "nll")
        }
    return out


# ---- stages -------------------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_new(path: Path, text: str) -> None:
    with open(path, "x") as handle:  # "x": never overwrite a result
        handle.write(text)


def check_tokenizers(tokenizers: Mapping[str, Any], texts: Sequence[str]) -> dict[str, Any]:
    """Token ids of every tokenizer on every text must be identical (``add_special_tokens=False``, as the harness)."""
    ref_name = next(iter(tokenizers))
    ref = [tokenizers[ref_name].encode(t, add_special_tokens=False) for t in texts]
    for name, tok in tokenizers.items():
        got = [tok.encode(t, add_special_tokens=False) for t in texts]
        if got != ref:
            bad = next(i for i, (a, b) in enumerate(zip(got, ref, strict=True)) if a != b)
            raise ValueError(f"tokenizer {name!r} differs from {ref_name!r} on held-out text {bad}")
    return {"identical": True, "tokenizers": list(tokenizers), "n_texts": len(texts), "add_special_tokens": False}


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[Sequence[int]],
    *,
    eos_ids: Sequence[int],
    pad_id: int,
    max_new_tokens: int,
    batch_size: int,
    device: Any,
) -> list[list[int]]:
    """Greedy continuation of every prompt (left-padded batches); each response ends at (and includes) the first eos."""
    from transformers import GenerationConfig

    gc = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=list(eos_ids),
        pad_token_id=int(pad_id),
        repetition_penalty=1.0,
    )
    out: list[list[int]] = []
    for start in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[start : start + batch_size]
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(chunk):
            ids[i, width - len(p) :] = torch.tensor(list(p))
            mask[i, width - len(p) :] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=gc)
        for i in range(len(chunk)):
            new = gen[i, width:].tolist()
            cut = next((k + 1 for k, t in enumerate(new) if t in set(eos_ids)), len(new))
            out.append(new[:cut])
    return out


def _peak_meta() -> dict[str, Any]:
    return {
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _run_meta(config_path: str, config_bytes: bytes, cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_bytes(config_bytes),
        "config": dict(cfg),
        "code_fingerprint": _code_fingerprint(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
    }


def _check_heldout(cfg: Mapping[str, Any], sink: Mapping[str, Any]) -> dict[str, Any]:
    docs = sink["heldout_docs"]
    expected = int(cfg.get("expected_n_docs", 141))
    if len(docs) != expected:
        raise ValueError(f"{len(docs)} held-out docs, expected {expected}")
    ours = holdout_sha256(sink["eval_samples"])
    ref = (sink["ref_summary"].get("calibration_provenance") or {}).get("holdout_sha256")
    if ours != ref:
        raise ValueError(f"held-out sha256 {ours} differs from the summary's calibration_provenance {ref}")
    return {"n_docs": len(docs), "holdout_sha256": ours, "summary_holdout_sha256": ref, "match": True}


def stage_generate(cfg: dict[str, Any], device: str, config_bytes: bytes, config_path: str) -> None:
    from merge_and_rebase.models.text_lm import TextLM

    from .llm import build_llm_inputs

    out_dir = Path(cfg["out_dir"])
    path = out_dir / RESPONSES_NAME
    if path.exists():
        raise FileExistsError(f"{path} exists; results are write-once")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sink: dict[str, Any] = {"context_only": True}
    list(build_llm_inputs(cfg, device, context_sink=sink))
    rt = sink["runtime"]
    holdout = _check_heldout(cfg, sink)
    docs = sink["heldout_docs"]
    texts = [d["text"] for d in docs]
    native_name = str(cfg.get("native_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"))
    native = TextLM.build(sink["build_cfg"](native_name))
    toks = check_tokenizers(
        {"source": rt.source_llm.tokenizer, "target": rt.target_llm.tokenizer, "native": native.tokenizer}, texts
    )
    tok = rt.target_llm.tokenizer
    prompt_ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    hashes_before = {"native": _state_sha(native.model), "target": _state_sha(rt.target_llm.model)}
    eos = native.model.generation_config.eos_token_id
    eos_ids = [int(eos)] if isinstance(eos, int) else [int(e) for e in eos]
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0])
    responses = greedy_generate(
        native.model,
        prompt_ids,
        eos_ids=eos_ids,
        pad_id=pad_id,
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        batch_size=int(cfg.get("batch_size", 4)),
        device=device,
    )
    hashes_after = {"native": _state_sha(native.model), "target": _state_sha(rt.target_llm.model)}
    lines = []
    for d, p, r in zip(docs, prompt_ids, responses, strict=True):
        lines.append(
            json.dumps(
                {
                    "doc_index": d["doc_index"],
                    "doc_key": d["key"],
                    "prompt_text": d["text"],
                    "prompt_ids": p,
                    "response_ids": r,
                    "response_text": tok.decode(r),
                },
                sort_keys=True,
            )
        )
    write_new(path, "\n".join(lines) + "\n")
    digest = sha256_file(path)
    write_new(out_dir / (RESPONSES_NAME + ".sha256"), digest + "\n")
    meta = {
        **_run_meta(config_path, config_bytes, cfg),
        "stage": "generate",
        "responses_sha256": digest,
        "holdout": holdout,
        "tokenizers": toks,
        "prompt_format": {
            "chat_template_applied": False,
            "add_special_tokens": False,
            "prompt": "task.doc_to_text(doc)",
        },
        "generation": {
            "greedy": True,
            "repetition_penalty": 1.0,
            "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
            "eos_ids": eos_ids,
            "pad_id": pad_id,
            "padding_side": "left",
        },
        "native_model": native_name,
        "n_hit_max_new_tokens": sum(1 for r in responses if not r or r[-1] not in set(eos_ids)),
        "state_sha256_before": hashes_before,
        "state_sha256_after": hashes_after,
        "models_unchanged": hashes_before == hashes_after,
        "seconds": time.perf_counter() - t0,
        **_peak_meta(),
    }
    write_new(out_dir / "generate.json", json.dumps(_jsonable(meta), indent=1, sort_keys=True))
    if hashes_before != hashes_after:
        raise GateError("a model changed during generation")
    print(f"[llm_readout] generated {len(responses)} responses, sha256 {digest}", flush=True)


def load_responses(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    path = out_dir / RESPONSES_NAME
    digest = sha256_file(path)
    recorded = (out_dir / (RESPONSES_NAME + ".sha256")).read_text().strip()
    if digest != recorded:
        raise ValueError(f"{path}: sha256 {digest} differs from the recorded {recorded}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()], digest


def _load_original(path: str, reference: Mapping[str, Tensor]) -> dict[str, Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in reference if k not in raw]
    if missing:
        raise KeyError(f"original_tv_path {path} lacks {len(missing)} keys, e.g. {missing[:2]}")
    return {k: raw[k].detach().float().clone() for k in reference}


def stage_measure(cfg: dict[str, Any], device: str, config_bytes: bytes, config_path: str) -> None:
    from merge_and_rebase.models.text_lm import TextLM

    from .llm import _load_reference_task_vector, build_llm_inputs

    out_dir = Path(cfg["out_dir"])
    measure_path = out_dir / "measure.json"
    if measure_path.exists():
        raise FileExistsError(f"{measure_path} exists; results are write-once")
    t0 = time.perf_counter()
    responses, resp_sha = load_responses(out_dir)
    sink: dict[str, Any] = {}
    gen = build_llm_inputs(cfg, device, context_sink=sink)
    inp = next(gen)
    gen.close()
    holdout = _check_heldout(cfg, sink)
    heldout = sink["heldout_docs"]
    if [d["doc_index"] for d in heldout] != [r["doc_index"] for r in responses] or [d["text"] for d in heldout] != [
        r["prompt_text"] for r in responses
    ]:
        raise ValueError("the responses file does not hold the held-out docs of the reference run, in order")
    rt = sink["runtime"]
    native_name = str(cfg.get("native_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"))
    native = TextLM.build(sink["build_cfg"](native_name))
    toks = check_tokenizers(
        {"source": rt.source_llm.tokenizer, "target": rt.target_llm.tokenizer, "native": native.tokenizer},
        [r["prompt_text"] for r in responses],
    )
    if [rt.target_llm.tokenizer.encode(r["prompt_text"], add_special_tokens=False) for r in responses] != [
        r["prompt_ids"] for r in responses
    ]:
        raise ValueError("stored prompt ids differ from the target tokenizer's")

    # -- the three updates (unit strength) ----------------------------------------------------------------------------
    ref_tv = inp.reference_task_vector
    keys = list(ref_tv)
    original = _load_original(str(cfg["original_tv_path"]), ref_tv)
    updates = [
        Update("original", original, float(inp.alpha_star), _task_vector_sha256(original), str(cfg["original_tv_path"]))
    ]
    summaries: dict[str, Any] = {"original": json.loads(Path(str(cfg["alpha_star_summary"])).read_text())}
    for name, summary_path in cfg["variant_runs"].items():
        summary = json.loads(Path(summary_path).read_text())
        alpha = float(summary["best_alpha"])
        merged = Path(summary_path).parent / "merged_state.pt"
        tv = {k: v / alpha for k, v in _load_reference_task_vector(str(merged), inp.target_base_state, keys).items()}
        updates.append(Update(name, tv, alpha, _task_vector_sha256(tv), str(merged)))
        summaries[name] = summary
    original_vs_reference = {
        "relative_frobenius_diff": float(
            (sum(float(((original[k].double() - ref_tv[k].double()) ** 2).sum()) for k in keys)) ** 0.5
            / max(sum(float((ref_tv[k].double() ** 2).sum()) for k in keys) ** 0.5, 1e-300)
        )
    }

    # -- the fit's own Procrustes maps, called exactly as runner.run_task does --------------------------------------------
    cfg_a = inp.config
    prepared = prepare_direct_residual_streaming(
        inp.source_base_model,
        inp.target_model,
        inp.source_loader,
        inp.target_loader,
        inp.pairing,
        num_batches=cfg_a.num_batches,
        seed=cfg_a.seed,
        device=inp.device,
        family_adapter=inp.family_adapter,
        source_ft_model=inp.source_ft_model,
        procrustes_source=cfg_a.procrustes_source,
        alignment_map=cfg_a.alignment_map,
        alignment_row_weighting=cfg_a.alignment_row_weighting,
        alignment_seed=cfg_a.alignment_seed,
        fidelity_alignment_diagnostics=bool(cfg_a.fidelity_holdout),
    )
    q = {j: prepared["q_by_position"][j].detach().cpu() for j in range(inp.pairing.target_depth)}
    q_calibration = {k: v for k, v in prepared["calibration"].items() if k != "indices"}
    del prepared

    setup = ReadoutSetup(
        device=torch.device(device) if isinstance(device, str) else device,
        fam=inp.family_adapter,
        target=inp.target_model,
        mounted=copy.deepcopy(inp.target_model),
        source_base=inp.source_base_model,
        source_ft=inp.source_ft_model,
        native=native.model.to(device).eval(),
        pairing=list(inp.pairing.pairing),
        q=q,
        target_base_state=inp.target_base_state,
        updates=updates,
        pad_id=int(rt.target_llm.tokenizer.pad_token_id or 0),
    )
    hashes_before = model_hashes(setup)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    record = run_measure(setup, responses, cfg)
    record["m4"]["spearman"] = attach_m4_scores(record["m4"]["rows"], summaries)
    hashes_after = model_hashes(setup)
    record["gates"]["models_unchanged"] = {
        "before": hashes_before,
        "after": hashes_after,
        "passed": hashes_before == hashes_after,
    }
    record["gates"]["calibration_matches_reference"] = {
        "passed": bool(inp.provenance.get("calibration_matches_reference"))
    }
    record["gates"]["tokenizers_identical"] = {"passed": bool(toks["identical"])}
    failed = [k for k, g in record["gates"].items() if g.get("passed") is False]
    record["failed_gates"] = failed
    record["provenance"] = {
        **_run_meta(config_path, config_bytes, cfg),
        "stage": "measure",
        "responses_sha256": resp_sha,
        "holdout": holdout,
        "tokenizers": toks,
        "split_doc_indices": {
            "selection": [responses[i]["doc_index"] for i in record["split"]["selection"]],
            "report": [responses[i]["doc_index"] for i in record["split"]["report"]],
        },
        "updates": {u.name: {"sha256": u.sha256, "best_alpha": u.best_alpha, "source": u.source} for u in updates},
        "original_vs_reference_task_vector": original_vs_reference,
        "q_calibration": q_calibration,
        "reference": inp.provenance,
    }
    write_new(measure_path, json.dumps(_jsonable(record), indent=1, sort_keys=True))
    run = {
        "seconds": time.perf_counter() - t0,
        "measure_seconds": record["seconds"],
        "failed_gates": failed,
        **_peak_meta(),
    }
    write_new(out_dir / "run.json", json.dumps(_jsonable(run), indent=1, sort_keys=True))
    print(f"[llm_readout] measure done in {run['seconds']:.0f}s; failed gates: {failed}", flush=True)
    if failed:
        raise GateError(f"correctness gate(s) failed: {failed}; see {measure_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["generate", "measure"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    stage: Callable[..., None] = stage_generate if args.stage == "generate" else stage_measure
    stage(cfg, args.device, config_bytes, args.config)


if __name__ == "__main__":
    main()
