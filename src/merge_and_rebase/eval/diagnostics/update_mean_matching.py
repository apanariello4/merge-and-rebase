"""Cross-task matching of the dataset MEAN SHIFT of two models (aggregator of the per-task ``update_kernel_cka`` runs).

Input: the run directories (``mean_vectors.pt`` + ``summary.json``) of ONE model pair (m source, k target) on several tasks
(the 8 Vision8 tasks; for the split A_B runs the merged directory). Each run holds, per block pair ``(i, j)`` and per
jackknife chunk of images, the T-vectors (anchor coordinates of the pairwise frame, identical for all tasks) summed over the
rows of the task's images: the encoded update ``dH P^T`` of model m with ITS expert of task t and of model k with ITS expert,
and the encoded base activations ``H_base P^T``. For a block, a row group (cls / patch / all) and a kind

* ``displacement``   v = mean encoded update;
* ``base_position``  v = mean encoded base activation (where the dataset lands; no expert involved);
* ``ft_position``    v = mean encoded fine-tuned activation (base + update),

the matrix ``C[t, t'] = cos(v_m^t, v_k^t')`` (anchor-centred vectors, as ``common_mode``) compares model m on dataset t with
model k on dataset t', each on its own task's images with its own expert. Reported per block / kind / group:

* ``row_excess[t] = C[t,t] - mean_{t' != t} C[t,t']`` (and the column version; the MEAN over tasks of the two is the same
  statistic, so only ``mean_row_excess`` is reported), each with a delete-one-chunk jackknife
  interval (``t_{G-1}``) and the mean over tasks. The chunk ``g`` of every task is deleted together (chunks paired by
  index): the chunks of different tasks hold different images, so the pairing is a convention that makes the 8 x 8 matrix
  one resampling unit; the interval measures the variability over image chunks of the whole matrix;
* the top-1 matching accuracy, ``mean_t [argmax_t' C[t,t'] == t]`` (row-wise) and ``mean_t' [argmax_t C[t,t'] == t']``
  (column-wise), with the exact permutation null over task labels (all ``n!`` matchings for ``n <= 8``, 20000 seeded
  random ones plus the identity otherwise): ``p = P(null >= observed)``; the same null for the mean excess.
* variants with the same layout: ``task_centred`` (the mean over tasks of each model removed before the cosine) and
  ``double_centred`` (row and column means removed from C); ``control_random_update`` (displacement, when the runs saved
  ``k_rand``): the matrix with model k replaced by its random update, and ``excess_minus_control`` with a paired jackknife;
  ``se_pairings``: the jackknife SE of the mean excess over 10 seeded random chunk pairings across tasks.

    python -m merge_and_rebase.eval.diagnostics.update_mean_matching --dirs D1 D2 ... --out OUT
    (or  python -m merge_and_rebase.eval.diagnostics.update_kernel_cka --matching D1 D2 ... --out OUT)

Writes ``OUT/matching.json`` and ``OUT/matching.md``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .update_kernel_cka import (
    MEAN_VECTORS_FILE,
    MS_GROUPS,
    MS_KINDS,
    jackknife_ci,
    ms_side_chunks,
)

Tensor = torch.Tensor
# the setting entries that must agree between the runs of one model pair (everything but the task and its experts)
SETTING_KEYS = (
    "state_dir", "frame", "T_stored", "anchor_filter", "anchor_filter_k", "pairing", "split", "split_seed", "batch_size",
    "grid_mode", "grid_reference", "grid_reference_tokens", "jackknife_chunks_actual", "sampling", "suite", "debiased",
    "compute_dtype",
)
TAGS = {"displacement": ("m_t", "k_t"), "base_position": ("m_t", "k_t"), "ft_position": ("m_t", "k_t")}
N_PERM_RANDOM = 20000


# ----------------------------------------------------------------------------------------------------------------
# statistics of one matrix
# ----------------------------------------------------------------------------------------------------------------


def corr_matrix(sum_m: Tensor, sum_k: Tensor, *, task_centre: bool = False, count_m: Tensor | None = None,
                count_k: Tensor | None = None) -> Tensor:
    """``C[t, t'] = cos(anchor-centred sum_m[t], anchor-centred sum_k[t'])`` for ``[n, T]`` sums (NaN for a zero vector).
    ``task_centre``: the mean vector over the tasks of each model is subtracted (after the anchor centring, from the MEAN
    vectors ``sum / count``, ``count`` the images per task; ones when not given) before the cosine."""
    def unit(x: Tensor, cnt: Tensor | None) -> Tensor:
        x = x.to(torch.float64)
        if task_centre:
            x = x / (torch.ones(x.shape[0], dtype=torch.float64) if cnt is None else cnt.to(torch.float64))[:, None]
        x = x - x.mean(1, keepdim=True)
        if task_centre:
            x = x - x.mean(0, keepdim=True)
        nrm = x.norm(dim=1, keepdim=True)
        return torch.where(nrm > 0, x / nrm.clamp_min(1e-300), torch.full_like(x, float("nan")))

    return unit(sum_m, count_m) @ unit(sum_k, count_k).T


def double_centre(c: Tensor) -> Tensor:
    """Row and column means removed from the matrix (grand mean added back)."""
    return c - c.mean(1, keepdim=True) - c.mean(0, keepdim=True) + c.mean()


def diag_excess(c: Tensor) -> tuple[Tensor, Tensor]:
    """``(row_excess, column_excess)``: the diagonal minus the mean of the off-diagonal entries of its row / column."""
    n = c.shape[0]
    d = c.diagonal()
    return d - (c.sum(1) - d) / (n - 1), d - (c.sum(0) - d) / (n - 1)


def top1(c: Tensor) -> tuple[Tensor, Tensor]:
    """``(row_arg, col_arg)``: ``argmax_t' C[t, t']`` for every row t and ``argmax_t C[t, t']`` for every column t'
    (NaN entries never win; an all-NaN row or column gets -1)."""
    filled = torch.nan_to_num(c, nan=-float("inf"))
    row_arg = filled.argmax(1)
    col_arg = filled.argmax(0)
    row_arg = torch.where(torch.isfinite(c).any(1), row_arg, torch.full_like(row_arg, -1))
    col_arg = torch.where(torch.isfinite(c).any(0), col_arg, torch.full_like(col_arg, -1))
    return row_arg, col_arg


def matchings(n: int, seed: int = 0) -> Tensor:
    """All permutations of ``n`` tasks (``n <= 8``; the identity first) or the identity + ``N_PERM_RANDOM`` seeded random
    ones: ``[P, n]``, row ``p`` pairs the row t of C with the column ``perms[p, t]``."""
    if n <= 8:
        return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    rnd = torch.stack([torch.randperm(n, generator=gen) for _ in range(N_PERM_RANDOM)])
    return torch.cat([torch.arange(n)[None], rnd])


def matching_stats(c: Tensor, perms: Tensor) -> dict[str, Any]:
    """Observed statistics of the matrix and their permutation nulls over task labels."""
    n = c.shape[0]
    ar = torch.arange(n)
    row_arg, col_arg = top1(c)
    acc_row, acc_col = float((row_arg == ar).double().mean()), float((col_arg == ar).double().mean())
    row_ex, col_ex = diag_excess(c)
    mean_row_ex = float(row_ex.mean())  # = the mean column excess (mean diagonal minus mean off-diagonal)
    # null: the matching pi pairs row t with column pi(t) (for the columns: column pi(t) with row t)
    hit_row = (row_arg[None] == perms).double().mean(1)
    hit_col = (col_arg[perms] == ar[None]).double().mean(1)
    sel = c[ar[None].expand_as(perms), perms]  # [P, n] C[t, pi(t)]
    ex_row = (sel - (c.sum(1)[None] - sel) / (n - 1)).mean(1)

    def summ(obs: float, null: Tensor, chance: float | None = None) -> dict[str, Any]:
        out = {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
               "p": float((null >= obs - 1e-12).double().mean()) if math.isfinite(obs) else float("nan"),
               "n_matchings": int(null.numel())}
        if chance is not None:
            out["chance"] = chance
        return out

    return {
        "top1_row": {**summ(acc_row, hit_row, 1.0 / n), "n_correct": int((row_arg == ar).sum()), "n": n},
        "top1_col": {**summ(acc_col, hit_col, 1.0 / n), "n_correct": int((col_arg == ar).sum()), "n": n},
        "mean_row_excess": summ(mean_row_ex, ex_row),
        "row_excess": row_ex, "col_excess": col_ex,
    }


# ----------------------------------------------------------------------------------------------------------------
# one block
# ----------------------------------------------------------------------------------------------------------------


def stack_tasks(parts_by_task: Sequence[Mapping[str, Mapping[str, Tensor]]], kind: str, group: str) -> tuple[Tensor, Tensor]:
    """``(vm, vk)`` ``[n_tasks, G, T]``: per-task, per-chunk vectors of model m and k (``m_t`` / ``k_t`` of every run)."""
    tx, ty = TAGS[kind]
    vm = torch.stack([ms_side_chunks(p, kind, tx, group) for p in parts_by_task])
    vk = torch.stack([ms_side_chunks(p, kind, ty, group) for p in parts_by_task])
    return vm, vk


VARIANTS = ("plain", "task_centred", "double_centred")
N_PAIRINGS = 10


def stack_counts(runs: Sequence[Mapping[str, Any]], block: int) -> Tensor:
    """Images per (task, chunk) ``[n, G]`` of a block (``n_images`` of the saved vectors)."""
    return torch.stack([torch.as_tensor(r["blocks"][block]["n_images"]).to(torch.float64) for r in runs])


def stack_control(parts_by_task: Sequence[Mapping[str, Mapping[str, Tensor]]], group: str) -> Tensor | None:
    """``[n, G, T]`` per-chunk displacement vectors of the random-update control (``k_rand``, draw 0) or None."""
    if not all("k_rand" in p for p in parts_by_task):
        return None
    return torch.stack([ms_side_chunks(p, "displacement", "k_rand", group) for p in parts_by_task])


def _cmat(vm: Tensor, vk: Tensor, cnt: Tensor, keep: Tensor | None, variant: str) -> Tensor:
    """The matrix of a variant from the chunks kept (``keep`` ``[n, G]`` bool; None = all)."""
    w = torch.ones(vm.shape[:2], dtype=torch.float64) if keep is None else keep.to(torch.float64)
    sm, sk, c = (w[:, :, None] * vm).sum(1), (w[:, :, None] * vk).sum(1), (w * cnt).sum(1)
    if variant == "task_centred":
        return corr_matrix(sm, sk, task_centre=True, count_m=c, count_k=c)
    c_plain = corr_matrix(sm, sk)
    return double_centre(c_plain) if variant == "double_centred" else c_plain


def _keep_masks(n: int, g: int, perm: Tensor | None) -> list[Tensor]:
    """The ``g`` delete-one masks ``[n, G]``: replicate ``d`` deletes chunk ``d`` of the first task and chunk
    ``perm[t, d]`` of task t (``perm`` ``[n, G]``, None = chunk d of every task: paired by index)."""
    out = []
    for d in range(g):
        keep = torch.ones(n, g, dtype=torch.bool)
        for t in range(n):
            keep[t, d if perm is None else int(perm[t, d])] = False
        out.append(keep)
    return out


def _variant_block(vm: Tensor, vk: Tensor, cnt: Tensor, tasks: Sequence[str], variant: str, perms: Tensor, *, ci: bool,
                   vc: Tensor | None, seed: int, n_pairings: int) -> dict[str, Any]:
    n, g, _ = vm.shape
    ar = torch.arange(n)
    full = _cmat(vm, vk, cnt, None, variant)
    st = matching_stats(full, perms)
    do_ci = ci and g >= 2
    masks = _keep_masks(n, g, None) if do_ci else []
    loo_c = [_cmat(vm, vk, cnt, m, variant) for m in masks]
    loo_ex = [diag_excess(c) for c in loo_c]

    def ent(theta: float, loo: Sequence[float]) -> dict[str, Any]:
        return {"value": theta, **(jackknife_ci(theta, loo) if loo else {})}

    out: dict[str, Any] = {
        "C": full.tolist(),
        "row_excess": {t: ent(float(st["row_excess"][i]), [float(x[0][i]) for x in loo_ex]) for i, t in enumerate(tasks)},
        # per-task column excess; its mean over tasks is the SAME number as the mean row excess (mean diag - mean off-diag)
        "col_excess": {t: ent(float(st["col_excess"][i]), [float(x[1][i]) for x in loo_ex]) for i, t in enumerate(tasks)},
        "mean_row_excess": {**st["mean_row_excess"], **ent(st["mean_row_excess"]["observed"], [float(x[0].mean()) for x in loo_ex])},
        "top1_row": dict(st["top1_row"]), "top1_col": dict(st["top1_col"]),
        "diag_mean": float(full.diagonal().mean()), "offdiag_mean": float((full.sum() - full.diagonal().sum()) / (n * (n - 1))),
    }
    if loo_c:  # the jackknife of a discrete accuracy is not an interval: the leave-one-chunk range only shows its stability
        for key, arg in (("top1_row", 0), ("top1_col", 1)):
            accs = [float((top1(c)[arg] == ar).double().mean()) for c in loo_c]
            out[key]["loo_range"] = [min(accs), max(accs)]
        # SE of the mean row excess over random chunk pairings across tasks (task 0 fixed, the others permuted)
        gen = torch.Generator().manual_seed(int(seed) * 7919 + 13)
        ses = []
        for _ in range(n_pairings):
            perm = torch.stack([torch.arange(g)] + [torch.randperm(g, generator=gen) for _ in range(n - 1)])
            vals = [float(diag_excess(_cmat(vm, vk, cnt, m, variant))[0].mean()) for m in _keep_masks(n, g, perm)]
            ses.append(jackknife_ci(st["mean_row_excess"]["observed"], vals)["se"])
        t_ses = torch.tensor(ses, dtype=torch.float64)
        out["se_pairings"] = {"n_pairings": n_pairings, "se_index_paired": out["mean_row_excess"].get("se"),
                              "mean": float(t_ses.mean()), "min": float(t_ses.min()), "max": float(t_ses.max()),
                              "sd": float(t_ses.std()) if n_pairings > 1 else float("nan")}
    if vc is not None:  # image-only control: model k replaced by its random update (same norm), same layout
        cc = _cmat(vm, vc, cnt, None, variant)
        cst = matching_stats(cc, perms)
        c_loo = [diag_excess(_cmat(vm, vc, cnt, m, variant))[0] for m in masks]
        ctl_mean = float(cst["mean_row_excess"]["observed"])
        diff = st["mean_row_excess"]["observed"] - ctl_mean
        out["control_random_update"] = {
            "C": cc.tolist(), "mean_row_excess": ent(ctl_mean, [float(x.mean()) for x in c_loo]),
            "top1_row": dict(cst["top1_row"]),
            "excess_minus_control": ent(diff, [float(a[0].mean() - b.mean()) for a, b in zip(loo_ex, c_loo, strict=True)]),
            "row_excess_minus_control": {t: ent(float(st["row_excess"][i] - cst["row_excess"][i]),
                                                [float(a[0][i] - b[i]) for a, b in zip(loo_ex, c_loo, strict=True)])
                                         for i, t in enumerate(tasks)},
        }
    return out


def block_matching(vm: Tensor, vk: Tensor, tasks: Sequence[str], *, ci: bool = True, seed: int = 0,
                   counts: Tensor | None = None, vc: Tensor | None = None, n_pairings: int = N_PAIRINGS) -> dict[str, Any]:
    """Matching of one (block, kind, group) from the per-chunk vectors ``[n, G, T]`` (chunks paired by index across the
    tasks). The top-level keys are the PLAIN variant; ``task_centred`` (the mean over the tasks of each model removed
    before the cosine) and ``double_centred`` (the matrix with row and column means removed) have the same layout.
    ``vc``: per-chunk vectors of the random-update control of model k (same layout, ``control_random_update`` keys).
    ``se_pairings``: jackknife SE of the mean row excess averaged over ``n_pairings`` seeded random chunk pairings."""
    n, g, _ = vm.shape
    if n < 2:
        raise ValueError("matching needs at least 2 tasks")
    cnt = torch.ones(n, g, dtype=torch.float64) if counts is None else counts.to(torch.float64)
    perms = matchings(n, seed)
    res = {v: _variant_block(vm, vk, cnt, tasks, v, perms, ci=ci, vc=vc, seed=seed, n_pairings=n_pairings) for v in VARIANTS}
    return {"tasks": list(tasks), "n_chunks": g, **res["plain"], "task_centred": res["task_centred"],
            "double_centred": res["double_centred"]}


# ----------------------------------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------------------------------


def load_run(run_dir: str | Path) -> dict[str, Any]:
    d = Path(run_dir)
    summary = json.loads((d / "summary.json").read_text())
    path = d / MEAN_VECTORS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: this run has no mean-shift vectors (rerun with mean_shift: true)")
    vec = torch.load(path, map_location="cpu", weights_only=False)
    return {"dir": str(d), "setting": summary["setting"], "meta": vec["meta"], "blocks": vec["blocks"]}


def check_runs(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Checks that the runs are the same model pair, frame and design on distinct tasks; returns the task names."""
    if len(runs) < 2:
        raise ValueError("matching needs at least 2 task runs (3 or more for a meaningful off-diagonal mean)")
    tasks = [r["meta"]["task"] for r in runs]
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"duplicate tasks: {tasks}")
    ref = runs[0]
    for r in runs[1:]:
        for key in SETTING_KEYS:
            if ref["setting"].get(key) != r["setting"].get(key):
                raise ValueError(f"{r['dir']}: setting {key!r} differs from {ref['dir']}")
        for side in ("source", "target"):
            a, b = ref["setting"][side], r["setting"][side]
            if (a["model"], a["pretrained"]) != (b["model"], b["pretrained"]):
                raise ValueError(f"{r['dir']}: {side} model differs")
        if sorted(ref["blocks"]) != sorted(r["blocks"]):
            raise ValueError(f"{r['dir']}: blocks {sorted(r['blocks'])} != {sorted(ref['blocks'])}")
        if ref["meta"]["n_chunks"] != r["meta"]["n_chunks"]:
            raise ValueError(f"{r['dir']}: number of jackknife chunks differs")
    return tasks


def matching_runs(runs: Sequence[Mapping[str, Any]], *, ci: bool = True, seed: int = 0) -> dict[str, Any]:
    tasks = check_runs(runs)
    ref = runs[0]
    blocks_out = []
    for j in sorted(ref["blocks"]):
        parts = [r["blocks"][j]["tags"] for r in runs]
        entry: dict[str, Any] = {"target_block": j, "source_block": ref["blocks"][j]["source_block"],
                                 "n_anchors": ref["blocks"][j]["n_anchors"], "kinds": {}}
        for kind in MS_KINDS:
            entry["kinds"][kind] = {}
            for group in MS_GROUPS:
                vm, vk = stack_tasks(parts, kind, group)
                entry["kinds"][kind][group] = block_matching(
                    vm, vk, tasks, ci=ci, seed=seed + j, counts=stack_counts(runs, j),
                    vc=stack_control(parts, group) if kind == "displacement" else None)
        blocks_out.append(entry)
    s0 = ref["setting"]
    setting = {
        "source": {"model": s0["source"]["model"], "pretrained": s0["source"]["pretrained"]},
        "target": {"model": s0["target"]["model"], "pretrained": s0["target"]["pretrained"]},
        "tasks": tasks, "run_dirs": [r["dir"] for r in runs],
        "experts": {r["meta"]["task"]: {"source": r["setting"]["source"]["expert"].get("path"),
                                        "target": r["setting"]["target"]["expert"].get("path")} for r in runs},
        "images": {r["meta"]["task"]: {k: r["setting"]["images"]["task"].get(k) for k in
                                       ("n_images", "image_indices_sha256", "labels_sha256", "n_split_total")} for r in runs},
        **{k: s0.get(k) for k in SETTING_KEYS}, "n_chunks": ref["meta"]["n_chunks"],
        "images_per_chunk": {r["meta"]["task"]: r["meta"]["images_per_chunk"] for r in runs},
        "n_matchings": int(matchings(len(tasks), seed).shape[0]), "ci": ci, "seed": seed,
        "ci_method": "delete-one jackknife over image chunks, t_{G-1}; chunk g of every task deleted together (paired by index)",
        "vector_definition": "anchor-centred sum over rows of the encoded update (displacement) / base activation "
                             "(base_position) / base + update (ft_position); cosine between model m on task t and model k on task t'",
    }
    return {"setting": setting, "blocks": blocks_out}


# ----------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------


def _f(x: Any) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _c(e: Mapping[str, Any] | None) -> str:
    return "" if not e else f"{_f(e.get('value'))} [{_f(e.get('lo'))}, {_f(e.get('hi'))}]"


def _acc(e: Mapping[str, Any]) -> str:
    return f"{e['n_correct']}/{e['n']} (p {_f(e['p'])})"


def render_matching_md(res: Mapping[str, Any]) -> str:
    s = res["setting"]
    blocks = res["blocks"]
    tasks = s["tasks"]
    lines = [
        f"# Matching del mean-shift fra task: m = {s['source']['model']}/{s['source']['pretrained']} (sorgente), "
        f"k = {s['target']['model']}/{s['target']['pretrained']} (target)", "",
        "## Domanda", "",
        "Lo spostamento medio del dataset (e la posizione in cui atterra) del modello m sul task t assomiglia a quello del modello k "
        "sullo STESSO task piu' che su un altro task? Misurato nelle coordinate delle ancore condivise del frame a coppie.", "",
        "## Procedura", "",
        f"- Task ({len(tasks)}): {', '.join(tasks)}. Ogni modello e' valutato sulle immagini del proprio task con il proprio esperto.",
        f"- Immagini per task: {', '.join(f'{t} {s['images'][t]['n_images']}' for t in tasks)} (split `{s['split']}`, seme {s['split_seed']}, "
        f"campionamento `{s['sampling']}`, batch {s['batch_size']}); chunk del jackknife: {s['n_chunks']} (immagini per chunk "
        f"{ {t: s['images_per_chunk'][t] for t in tasks} }).",
        f"- Ancore: state_dir `{s['state_dir']}`, T = {s['T_stored']}, filtro `{s['anchor_filter']}` k={s['anchor_filter_k']}; "
        f"accoppiamento blocchi (j -> i): {s['pairing']}.",
        "- Vettori: somma sulle righe (CLS / patch / all) di dH P^T (`displacement`), H_base P^T (`base_position`), H_base P^T + dH P^T "
        "(`ft_position`), per chunk di immagini; C[t, t'] = coseno (centrato sulle ancore) fra il vettore di m sul task t e quello di k "
        "sul task t'.",
        "- Statistiche per blocco: eccesso di riga = C[t,t] - media dei C[t,t'] con t' != t (e di colonna), IC95 jackknife delete-one "
        "(t_(G-1)) cancellando lo stesso indice di chunk in tutti i task (accoppiati per indice: i chunk di task diversi hanno immagini "
        "diverse, e' una convenzione di ricampionamento); accuratezza top-1 per riga (argmax_t' C[t,t'] == t) e per colonna, con "
        f"null di permutazione esatto sulle etichette dei task ({s['n_matchings']} abbinamenti); chance = 1/{len(tasks)}.",
        "- Esperti: " + "; ".join(f"{t}: m {s['experts'][t]['source']}, k {s['experts'][t]['target']}" for t in tasks) + ".",
        f"- Directory dei run: {', '.join(s['run_dirs'])}.", "",
        "## Risultato", "",
    ]
    titles = {
        "displacement": "displacement (PRIMARIA: spostamento medio dell'update dH)",
        "base_position": "base_position (riferimento SOLO-IMMAGINI: nessun esperto, dipende dal modello e dalle immagini)",
        "ft_position": "ft_position (SECONDARIA: dominata dalla posizione base)",
    }
    for kind in ("displacement", "base_position", "ft_position"):
        for group in ("all", "patch", "cls"):
            note = " (con 256 patch su 257 righe `all` coincide di fatto con `patch`)" if group == "all" else ""
            lines += [
                f"### {titles[kind]}, gruppo {group}{note}", "",
                "Eccesso medio = media sui task di C[t,t] meno la media dei C[t,t'] fuori diagonale (e' anche l'eccesso di colonna medio: "
                "stessa statistica). IC95 jackknife; SE(10 abbinamenti) = SE jackknife medio su 10 abbinamenti casuali dei chunk fra task "
                "[min, max].", "",
                "| blocco j (i) | eccesso medio [IC] | p | top-1 riga | top-1 col | SE (10 abbinamenti) | task-centrato: eccesso [IC] | p | top-1 riga | "
                "doppio-centrato: eccesso | controllo casuale: eccesso [IC] | eccesso - controllo [IC] | diag / fuori diag |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for b in blocks:
                e = b["kinds"][kind][group]
                tc, dc = e["task_centred"], e["double_centred"]
                ctl = e.get("control_random_update")
                sp = e.get("se_pairings")
                lines.append(
                    f"| {b['target_block']} ({b['source_block']}) | {_c(e['mean_row_excess'])} | {_f(e['mean_row_excess']['p'])} | "
                    f"{_acc(e['top1_row'])} | {_acc(e['top1_col'])} | "
                    f"{'' if not sp else f'{_f(sp['mean'])} [{_f(sp['min'])}, {_f(sp['max'])}]'} | "
                    f"{_c(tc['mean_row_excess'])} | {_f(tc['mean_row_excess']['p'])} | {_acc(tc['top1_row'])} | "
                    f"{_f(dc['mean_row_excess']['value'])} | {'' if not ctl else _c(ctl['mean_row_excess'])} | "
                    f"{'' if not ctl else _c(ctl['excess_minus_control'])} | {_f(e['diag_mean'])} / {_f(e['offdiag_mean'])} |"
                )
            lines.append("")
    shown = sorted({blocks[0]["target_block"], blocks[len(blocks) // 2]["target_block"], blocks[-1]["target_block"]})
    for kind in ("displacement", "ft_position"):
        for b in blocks:
            if b["target_block"] not in shown:
                continue
            for label, e in (("semplice", b["kinds"][kind]["all"]), ("task-centrata", b["kinds"][kind]["all"]["task_centred"]),
                             ("doppio-centrata", b["kinds"][kind]["all"]["double_centred"])):
                lines += [f"### Matrice C {label}, {kind}, gruppo all, blocco {b['target_block']} (righe: m sul task t, colonne: k sul task t')", "",
                          "| t \\ t' | " + " | ".join(tasks) + " | eccesso riga [IC] |", "|---|" + "---|" * (len(tasks) + 1)]
                for i, t in enumerate(tasks):
                    lines.append(f"| {t} | " + " | ".join(_f(x) for x in e["C"][i]) + f" | {_c(e['row_excess'][t])} |")
                lines.append("")
    lines += [
        "## Come si legge", "",
        "- Eccesso di riga > 0 con IC lontano da 0 e top-1 sopra la chance (p piccolo): il mean-shift di m e k e' specifico del task. "
        "Per `base_position` non interviene alcun esperto: misura se i due modelli vedono lo stesso dataset nello stesso modo.",
        "- Il p del null permuta le etichette dei task: vale per l'ipotesi 'nessun legame fra identita' del task e abbinamento'; "
        "i blocchi non sono indipendenti (blocchi consecutivi condividono la sorgente e le immagini).",
        "- `displacement` e' la misura primaria. `base_position` e' il riferimento solo-immagini (nessun esperto): quanto i due modelli "
        "concordano sul dataset gia' prima del fine-tuning. `ft_position` (base + update) e' SECONDARIA: e' dominata dalla base. Con 256 patch su "
        "257 righe il gruppo `all` e' di fatto `patch`.",
        "- Variante task-centrata: da ciascun modello si toglie il vettore medio sui task (dopo il centraggio sulle ancore) prima del coseno; "
        "rimuove la direzione comune a tutti i task. La matrice doppio-centrata toglie medie di riga e colonna da C.",
        "- Controllo casuale (solo displacement, se i run hanno `k_rand`): stessa matrice con k sostituito dall'update casuale di uguale "
        "norma (estrazione 0, stesse immagini del task); `eccesso - controllo` ha IC jackknife appaiato (stessi chunk cancellati).",
        "- SE con 10 abbinamenti: lo schema di accoppiamento dei chunk fra task e' una convenzione; l'SE jackknife e' ricalcolato per 10 "
        "abbinamenti casuali con seme (task 0 fisso) e se ne mostrano media e intervallo.",
        "- Il `loo_range` di top-1 nel JSON e' solo un indicatore di stabilita' (il jackknife di una statistica discreta non e' un IC).", "",
        "## File", "", "- `matching.json` (matrici C complete, eccessi per riga con IC, accuratezze, null), `matching.md`.", "",
    ]
    return "\n".join(lines)


def run_matching(dirs: Sequence[str | Path], out_dir: str | Path, *, ci: bool = True, seed: int = 0) -> dict[str, Any]:
    res = matching_runs([load_run(d) for d in dirs], ci=ci, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matching.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "matching.md").write_text(render_matching_md(res))
    return res


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dirs", nargs="+", required=True, metavar="DIR", help="per-task run directories of one model pair")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0, help="seed of the random matchings (only used for more than 8 tasks)")
    args = parser.parse_args(argv)
    run_matching(args.dirs, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
