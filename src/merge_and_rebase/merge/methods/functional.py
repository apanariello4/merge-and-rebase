from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


def _parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = str(name).strip().lower()
    if key.startswith("torch."):
        key = key[len("torch.") :]
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if key not in mapping:
        raise ValueError("Unknown dtype. Use one of: fp16, bf16, fp32, fp64 (or float16/32/64).")
    return mapping[key]


def _default_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
    if weights is None:
        return torch.ones(n, dtype=torch.float32)
    if len(weights) != n:
        raise ValueError("weights length must match number of matrices")
    return torch.tensor([float(w) for w in weights], dtype=torch.float32)


def _merge_method_params(method_params: Mapping[str, Any] | None, technical_params: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(method_params or {})
    out.update(dict(technical_params))
    return out


def _validate_matrices(matrices: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    mats = list(matrices)
    if not mats:
        raise ValueError("At least one matrix is required.")
    ref_shape = tuple(mats[0].shape)
    for i, m in enumerate(mats):
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"Matrix #{i} is not a torch.Tensor.")
        if tuple(m.shape) != ref_shape:
            raise ValueError(f"All matrices must have the same shape. got {tuple(m.shape)} vs {ref_shape}")
    return mats


def _require_2d(mats: Sequence[torch.Tensor], method_name: str) -> None:
    if mats[0].ndim != 2:
        raise ValueError(f"{method_name} requires 2D matrices. got shape {tuple(mats[0].shape)}")


def _rank_from_singular_values(num_singular_values: int, *, sv_reduction: float, max_rank: int | None) -> int:
    r = max(1, int(num_singular_values * float(sv_reduction)))
    if max_rank is not None:
        r = min(r, int(max_rank))
    return max(1, int(r))


def _stack_flatten(mats: Sequence[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    rows = [m.reshape(-1).to(dtype=dtype) for m in mats]
    return torch.stack(rows, dim=0)


def _task_arithmetic_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    _ = params
    out = torch.zeros_like(matrices[0])
    for wi, m in zip(w, matrices, strict=True):
        out = out + float(wi) * m.to(dtype=out.dtype, device=out.device)
    return out


def _weighted_average_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    normalize = str(params.get("normalize", "sumw"))
    if normalize == "sumw":
        denom = float(w.sum().clamp_min(1e-12).item())
    elif normalize == "n":
        denom = float(len(matrices))
    else:
        raise ValueError("normalize must be 'sumw' or 'n'")

    out = torch.zeros_like(matrices[0])
    for wi, m in zip(w, matrices, strict=True):
        out = out + float(wi) * m.to(dtype=out.dtype, device=out.device)
    return out / max(1e-12, denom)


def _tsv_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("tsv_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return _weighted_average_impl(matrices, w, {"normalize": "sumw"})
    _require_2d(matrices, "tsv_merge")

    sv_reduction = float(params.get("sv_reduction", 1.0 / max(1, len(matrices))))
    if not (0.0 < sv_reduction <= 1.0):
        raise ValueError("tsv_merge method_params['sv_reduction'] must be in (0, 1].")

    max_rank_raw = params.get("max_rank", None)
    max_rank = None if max_rank_raw is None else int(max_rank_raw)
    if max_rank is not None and max_rank <= 0:
        raise ValueError("tsv_merge method_params['max_rank'] must be > 0.")

    svd_dtype = _parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("tsv_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")
    accum_dtype = _parse_dtype(str(params.get("accum_dtype", "float32")))

    ref = matrices[0]
    k_min = min(int(ref.shape[0]), int(ref.shape[1]))
    r = _rank_from_singular_values(k_min, sv_reduction=sv_reduction, max_rank=max_rank)
    n_tasks = len(matrices)
    rt = r * n_tasks

    sum_u = torch.zeros((int(ref.shape[0]), rt), dtype=accum_dtype, device="cpu")
    sum_s = torch.zeros((rt,), dtype=accum_dtype, device="cpu")
    sum_v = torch.zeros((rt, int(ref.shape[1])), dtype=accum_dtype, device="cpu")

    for i, m in enumerate(matrices):
        mat = m.detach().to(device="cpu", dtype=svd_dtype)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        lo = i * r
        hi = lo + r
        sum_u[:, lo:hi] = u[:, :r].to(dtype=accum_dtype, device="cpu")
        sum_s[lo:hi] = (s[:r] * float(w[i])).to(dtype=accum_dtype, device="cpu")
        sum_v[lo:hi, :] = vh[:r, :].to(dtype=accum_dtype, device="cpu")

    u_u, _, vh_u = torch.linalg.svd(sum_u.to(dtype=svd_dtype), full_matrices=False)
    u_v, _, vh_v = torch.linalg.svd(sum_v.to(dtype=svd_dtype), full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, vh_u, torch.diag(sum_s.to(dtype=svd_dtype)), u_v, vh_v))
    return merged.to(dtype=ref.dtype, device=ref.device)


def _isoc_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("isoc_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return _weighted_average_impl(matrices, w, {"normalize": "sumw"})
    _require_2d(matrices, "isoc_merge")
    svd_dtype = _parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("isoc_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")

    combined = torch.zeros_like(matrices[0])
    for wi, m in zip(w, matrices, strict=True):
        combined = combined + float(wi) * m.to(dtype=combined.dtype, device=combined.device)

    u, s, vh = torch.linalg.svd(combined.to(dtype=svd_dtype), full_matrices=False)
    if s.numel() == 0:
        return torch.zeros_like(combined)
    s_iso = torch.ones_like(s) * s.mean()
    out = u @ torch.diag(s_iso) @ vh
    return out.to(dtype=combined.dtype, device=combined.device)


def _isocts_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if matrices[0].ndim == 1:
        vector_1d_merge = str(params.get("vector_1d_merge", "zero")).strip().lower()
        if vector_1d_merge not in {"zero", "average"}:
            raise ValueError("isocts_merge method_params['vector_1d_merge'] must be 'zero' or 'average'.")
        if vector_1d_merge == "zero":
            return torch.zeros_like(matrices[0])
        return _weighted_average_impl(matrices, w, {"normalize": "sumw"})
    _require_2d(matrices, "isocts_merge")
    common_space_fraction = float(params.get("common_space_fraction", 0.8))
    svd_dtype = _parse_dtype(str(params.get("svd_dtype", "float64")))
    if svd_dtype not in {torch.float32, torch.float64}:
        raise ValueError("isocts_merge method_params['svd_dtype'] must be float32/fp32 or float64/fp64.")

    ref = matrices[0]
    mats = [m.to(dtype=svd_dtype) for m in matrices]
    combined_w = sum(float(wi) * m for wi, m in zip(w, mats, strict=True))

    n_tasks = len(mats)
    min_dim = min(combined_w.shape)
    if min_dim == 0:
        return torch.zeros_like(ref)

    common_space_dim = int(min_dim * common_space_fraction)
    common_space_dim = max(0, min(common_space_dim, min_dim))
    task_specific_total_dim = max(0, min_dim - common_space_dim)
    task_dims_per_task = int(task_specific_total_dim // max(1, n_tasks))
    task_specific_total_dim = task_dims_per_task * n_tasks
    common_space_dim = min_dim - task_specific_total_dim

    u, s, vh = torch.linalg.svd(combined_w, full_matrices=False)
    common_u = u[:, :common_space_dim]
    common_s = s[:common_space_dim]
    common_v = vh[:common_space_dim, :]

    combined_space_u = torch.zeros_like(u)
    combined_space_s = torch.zeros_like(s)
    combined_space_v = torch.zeros_like(vh)

    if common_space_dim > 0:
        common_proj = common_u @ common_u.T
    else:
        common_proj = torch.zeros((combined_w.shape[0], combined_w.shape[0]), dtype=svd_dtype, device=combined_w.device)

    for task_idx, mat in enumerate(mats):
        mat_task_space = mat - (common_proj @ mat)
        u_ts, s_ts, vh_ts = torch.linalg.svd(mat_task_space, full_matrices=False)

        start = task_idx * task_dims_per_task
        end = (task_idx + 1) * task_dims_per_task
        if task_dims_per_task > 0:
            combined_space_u[:, start:end] = u_ts[:, :task_dims_per_task]
            combined_space_s[start:end] = s_ts[:task_dims_per_task]
            combined_space_v[start:end, :] = vh_ts[:task_dims_per_task, :]

    common_start = n_tasks * task_dims_per_task
    common_end = common_start + common_space_dim
    if common_space_dim > 0:
        combined_space_u[:, common_start:common_end] = common_u
        combined_space_s[common_start:common_end] = common_s
        combined_space_v[common_start:common_end, :] = common_v

    u_u, _, vh_u = torch.linalg.svd(combined_space_u, full_matrices=False)
    u_v, _, vh_v = torch.linalg.svd(combined_space_v, full_matrices=False)
    ortho_u = u_u @ vh_u
    ortho_v = u_v @ vh_v

    if combined_space_s.numel() > 0:
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

    out = ortho_u @ torch.diag(combined_space_s) @ ortho_v
    return out.to(dtype=ref.dtype, device=ref.device)


def _dare_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    if "drop_rate" in params:
        drop_rate = float(params["drop_rate"])
    elif "p" in params:
        drop_rate = float(params["p"])
    elif "keep_ratio" in params:
        drop_rate = 1.0 - float(params["keep_ratio"])
    else:
        drop_rate = 0.9

    if not (0.0 <= drop_rate < 1.0):
        raise ValueError("drop_rate must satisfy 0 <= drop_rate < 1.")

    seed_val = params.get("seed", None)
    seed = None if seed_val is None else int(seed_val)
    rescale = bool(params.get("rescale", True))
    work_dtype = _parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    flat = _stack_flatten(matrices, dtype=work_dtype)
    keep_prob = 1.0 - float(drop_rate)

    if keep_prob == 1.0:
        sparse = flat
    else:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=flat.device)
            gen.manual_seed(seed)
        mask = (torch.rand(flat.shape, device=flat.device, generator=gen) < keep_prob).to(flat.dtype)
        sparse = flat * mask
        if rescale:
            sparse = sparse / keep_prob

    merged_flat = (sparse * w.to(device=flat.device, dtype=flat.dtype).view(-1, 1)).sum(dim=0)
    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


def _topk_mask(M: torch.Tensor, topk: float) -> tuple[torch.Tensor, torch.Tensor]:
    if topk > 1.0:
        topk = topk / 100.0
    topk = float(topk)

    if topk >= 1.0:
        mask = torch.ones_like(M, dtype=torch.bool)
        return M, mask

    _, d = M.shape
    k = max(1, int(d * topk))
    vals, _ = torch.topk(M.abs(), k=k, dim=1, largest=True, sorted=False)
    thr = vals.min(dim=1, keepdim=True).values
    mask = M.abs() >= thr
    return M * mask, mask


def _resolve_sign(M: torch.Tensor) -> torch.Tensor:
    if torch.all(M == 0):
        return torch.ones(M.shape[1], device=M.device, dtype=torch.float32)
    s = torch.sign(M.sum(dim=0))
    global_majority = torch.sign(s.sum())
    global_majority = global_majority if global_majority != 0 else torch.tensor(1.0, device=s.device)
    s[s == 0] = global_majority
    return s


def _disjoint_merge(M: torch.Tensor, ref_sign: torch.Tensor, *, w: torch.Tensor, merge: str) -> torch.Tensor:
    keep = torch.where(ref_sign.unsqueeze(0) > 0, M > 0, M < 0)
    selected = M * keep

    w_row = w.to(selected.device, selected.dtype).view(-1, 1)
    selected = selected * w_row

    if merge == "mean":
        denom = (keep.to(selected.dtype) * w_row).sum(dim=0).clamp_min(1e-12)
        return selected.sum(dim=0) / denom
    if merge == "sum":
        return selected.sum(dim=0)
    if merge == "max":
        vals, _ = selected.abs().max(dim=0)
        return vals * ref_sign.to(vals.dtype)
    raise ValueError(f"Unknown TIES merge type '{merge}'")


def _ties_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    merging_type = str(params.get("merging_type", "mean"))
    topk = float(params.get("topk", 1.0))
    work_dtype = _parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    flat = _stack_flatten(matrices, dtype=work_dtype)
    pruned, _mask = _topk_mask(flat, topk=topk)
    sign = _resolve_sign(pruned)
    merged_flat = _disjoint_merge(pruned, sign, w=w, merge=merging_type)
    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


def _validate_ratios(*, clamp_min_ratio: float, clamp_max_ratio: float, att_ratio: float) -> None:
    if not (0.0 <= clamp_min_ratio < 1.0):
        raise ValueError("clamp_min_ratio must be in [0, 1).")
    if not (0.0 <= clamp_max_ratio < 1.0):
        raise ValueError("clamp_max_ratio must be in [0, 1).")
    if clamp_min_ratio + clamp_max_ratio >= 1.0:
        raise ValueError("clamp_min_ratio + clamp_max_ratio must be < 1.")
    if not (0.0 < att_ratio <= 1.0):
        raise ValueError("att_ratio must be in (0, 1].")


def _normalize_minmax(x: torch.Tensor, *, dim: int, eps: float = 1e-12) -> torch.Tensor:
    min_values = x.amin(dim=dim, keepdim=True)
    max_values = x.amax(dim=dim, keepdim=True)
    denom = (max_values - min_values).clamp_min(eps)
    return (x - min_values) / denom


def _clamp_by_ratio(x: torch.Tensor, *, min_ratio: float, max_ratio: float) -> torch.Tensor:
    if x.ndim == 1:
        d = x.shape[0]
        sorted_x, _ = torch.sort(x)
        lo_idx = int(d * min_ratio)
        hi_idx = int(d * (1.0 - max_ratio) - 1)
        hi_idx = max(lo_idx, hi_idx)
        min_v = sorted_x[lo_idx]
        max_v = sorted_x[hi_idx]
        return torch.clamp(x, min=min_v, max=max_v)

    if x.ndim == 2:
        d = x.shape[1]
        sorted_x, _ = torch.sort(x, dim=1)
        lo_idx = int(d * min_ratio)
        hi_idx = int(d * (1.0 - max_ratio) - 1)
        hi_idx = max(lo_idx, hi_idx)
        min_v = sorted_x[:, lo_idx].unsqueeze(1)
        max_v = sorted_x[:, hi_idx].unsqueeze(1)
        return torch.clamp(x, min=min_v, max=max_v)

    raise ValueError(f"Expected x to be 1D or 2D, got shape {tuple(x.shape)}")


def _pcb_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    clamp_min_ratio = float(params.get("clamp_min_ratio", 0.01))
    clamp_max_ratio = float(params.get("clamp_max_ratio", 0.01))
    att_ratio = float(params.get("att_ratio", 0.05))
    lam = float(params.get("lam", 1.2))

    _validate_ratios(
        clamp_min_ratio=clamp_min_ratio,
        clamp_max_ratio=clamp_max_ratio,
        att_ratio=att_ratio,
    )

    work_dtype = _parse_dtype(str(params.get("work_dtype", "float32")))

    ref = matrices[0]
    M = _stack_flatten(matrices, dtype=work_dtype)

    abs_M = M.abs()
    abs_clamped = _clamp_by_ratio(abs_M, min_ratio=clamp_min_ratio, max_ratio=clamp_max_ratio)
    clamped_M = M.sign() * abs_clamped

    norm_abs = _normalize_minmax(abs_clamped, dim=1)
    intra = torch.exp(float(M.shape[0]) * norm_abs.square())
    signed_norm = M.sign() * norm_abs
    inter = torch.tanh(M * signed_norm.sum(dim=0))
    balancing = intra * inter

    scale_seed = _clamp_by_ratio(balancing, min_ratio=1.0 - att_ratio, max_ratio=0.0)
    scale = _normalize_minmax(scale_seed, dim=1)

    lams = (float(lam) * w.to(device=M.device, dtype=M.dtype)).view(-1, 1)
    num = (clamped_M * lams * scale).sum(dim=0)
    den = scale.sum(dim=0).clamp_min(1e-12)
    merged_flat = num / den

    return merged_flat.view_as(ref).to(dtype=ref.dtype, device=ref.device)


def _cart_merge_impl(
    matrices: list[torch.Tensor],
    w: torch.Tensor,
    params: Mapping[str, Any],
) -> torch.Tensor:
    _require_2d(matrices, "cart_merge")

    pruning_rank = float(params.get("pruning_rank", 4))
    scaling_coeffs = float(params.get("scaling_coeffs", 0.5))

    theta_avg = torch.stack(matrices).mean(dim=0)
    sum_term = torch.zeros_like(theta_avg)

    for i, mat in enumerate(matrices):
        tau = mat - theta_avg
        u, s, vh = torch.linalg.svd(tau.to(torch.float64), full_matrices=False)
        rank_k = int(math.ceil(float(pruning_rank) * float(s.shape[0])))
        rank_k = max(1, min(int(s.shape[0]), rank_k))
        recon = u[:, :rank_k] @ torch.diag(s[:rank_k]) @ vh[:rank_k, :]
        sum_term = sum_term + recon.to(dtype=theta_avg.dtype, device=theta_avg.device) * float(w[i])

    return theta_avg + float(scaling_coeffs) * sum_term


_IMPLS: dict[str, Callable[[list[torch.Tensor], torch.Tensor, Mapping[str, Any]], torch.Tensor]] = {
    "task_arithmetic": _task_arithmetic_impl,
    "weighted_average": _weighted_average_impl,
    "tsv_merge": _tsv_merge_impl,
    "isoc_merge": _isoc_merge_impl,
    "isocts_merge": _isocts_merge_impl,
    "dare_merge": _dare_merge_impl,
    "ties_merge": _ties_merge_impl,
    "pcb": _pcb_merge_impl,
    "cart_merge": _cart_merge_impl,
}

_ALIASES: dict[str, str] = {
    "pcb_merge": "pcb",
}


def list_functional_methods() -> list[str]:
    return sorted(set(_IMPLS.keys()) | set(_ALIASES.keys()))


def merge_functional(
    method_name: str,
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    canonical = _ALIASES.get(method_name, method_name)
    if canonical not in _IMPLS:
        raise KeyError(f"Unknown functional merge method '{method_name}'. Available: {list_functional_methods()}")

    mats = _validate_matrices(matrices)
    w = _default_weights(len(mats), weights)
    params = _merge_method_params(method_params, technical_params)

    merged = _IMPLS[canonical](mats, w, params)
    return (float(alpha) * merged).to(dtype=mats[0].dtype, device=mats[0].device)


def merge_task_arithmetic(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "task_arithmetic",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_weighted_average(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "weighted_average",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_tsv(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "tsv_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_isoc(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "isoc_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_isocts(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "isocts_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_dare(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "dare_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_ties(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "ties_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_pcb(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "pcb",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_cart(
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        "cart_merge",
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def merge_raw_matrices(
    method_name: str,
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        method_name,
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )
