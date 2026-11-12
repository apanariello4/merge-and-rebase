"""Task-data-free calibration source 2: model inversion from the SOURCE fine-tuned model (DeepInversion-style).

Pixels are optimised in the CLIP-normalised space from Gaussian noise (seeded) so that the teacher (source fine-tuned
CLIP + its zero-shot head of the task's classes) classifies each image as its target class. Same recipe as the
reviewed ``dcvit_compress.generate_synthetic``: Adam (betas 0.5/0.9), peak lr 0.25 (linear warm-up, cosine decay),
random roll jitter (+-30 px) and flip per iteration, loss ``CE + l2 * |x| + tv_l2 * TV2`` (no BatchNorm term for ViTs),
clamp to the valid pixel range [0, 1] after de-normalisation. bf16 autocast is optional (synthesis only).

``class_targets`` balances the classes round-robin (C > n_images: ``n_images`` classes drawn with the seed).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def class_targets(num_classes: int, n_images: int, seed: int) -> tuple[list[int], list[int]]:
    """Target class of each image, balanced round-robin over a seed-permuted class order, then shuffled.

    Returns (targets in stored order, the classes used). Counts per class differ by at most one; with
    ``num_classes > n_images`` the classes are a seeded sample of size ``n_images`` (one image each).
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_classes, generator=g).tolist()
    classes = sorted(perm[:n_images]) if num_classes > n_images else list(range(num_classes))
    visit = [classes[i] for i in torch.randperm(len(classes), generator=g).tolist()]
    targets = [visit[i % len(visit)] for i in range(n_images)]
    shuffle = torch.randperm(n_images, generator=g).tolist()
    return [targets[i] for i in shuffle], classes


def image_prior(x: torch.Tensor, beta: float = 0.6) -> torch.Tensor:
    """TV-l2 of DeepInversion: sum over four neighbour differences of ``|d|_2 ** beta``."""
    d1, d2 = x[:, :, :, :-1] - x[:, :, :, 1:], x[:, :, :-1, :] - x[:, :, 1:, :]
    d3, d4 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:], x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    return sum(torch.norm(d) ** beta for d in (d1, d2, d3, d4))


def invert(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    targets: Sequence[int],
    *,
    iters: int,
    seed: int,
    mean: Sequence[float],
    std: Sequence[float],
    image_size: int = 224,
    device: str | torch.device = "cpu",
    lr: float = 0.25,
    jitter: int = 30,
    flip: bool = True,
    l2: float = 1e-5,
    tv_l2: float = 1e-4,
    beta: float = 0.6,
    warmup: int = 100,
    micro_batch: int = 16,
    amp_dtype: torch.dtype | None = None,
    log_every: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[torch.Tensor, list[float]]:
    """Optimise ``len(targets)`` images against ``logits_fn`` (frozen teacher, ``[B,3,H,W] -> [B,C]``).

    Returns (images ``[n,3,H,W]`` float32 CPU in the normalised space, CE trace of the optimised batch). The jitter
    and flip are shared by the whole batch within an iteration (as in the reference implementation); the gradient is
    accumulated over micro-batches so peak GPU memory is independent of ``len(targets)``.
    """
    dev = torch.device(device)
    n = len(targets)
    tgt = torch.tensor(list(targets), dtype=torch.long, device=dev)
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(n, 3, image_size, image_size, generator=g).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, betas=(0.5, 0.9), eps=1e-8)
    m_t = torch.tensor(list(mean), device=dev).view(1, 3, 1, 1)
    s_t = torch.tensor(list(std), device=dev).view(1, 3, 1, 1)
    lo, hi = (0 - m_t) / s_t, (1 - m_t) / s_t
    rng = random.Random(int(seed))
    trace: list[float] = []
    use_amp = amp_dtype is not None and dev.type == "cuda"
    for it in range(iters):
        if it < warmup:
            cur = lr * (it + 1) / warmup
        else:
            cur = 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(iters - warmup, 1))) * lr
        for grp in opt.param_groups:
            grp["lr"] = cur
        off1, off2 = rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)
        jit = torch.roll(x, shifts=(off1, off2), dims=(2, 3))
        if flip and rng.random() > 0.5:
            jit = torch.flip(jit, dims=(3,))
        opt.zero_grad(set_to_none=True)
        prior = tv_l2 * image_prior(jit, beta) + l2 * torch.norm(jit.reshape(n, -1), dim=1).mean()
        prior.backward(retain_graph=True)
        jd, grad = jit.detach(), torch.zeros_like(jit)
        ce_total = 0.0
        for s in range(0, n, micro_batch):
            xm = jd[s : s + micro_batch].clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = logits_fn(xm)
            ce = F.cross_entropy(logits.float(), tgt[s : s + micro_batch], reduction="sum") / n
            ce.backward()
            grad[s : s + micro_batch] = xm.grad
            ce_total += float(ce.detach())
        jit.backward(grad)
        opt.step()
        with torch.no_grad():
            x.copy_(torch.max(torch.min(x, hi), lo))
        trace.append(ce_total)
        if log_every and (it % log_every == 0 or it == iters - 1):
            log(f"  inversion it {it + 1}/{iters} CE={ce_total:.4f} lr={cur:.4f}")
    return x.detach().cpu(), trace


@torch.no_grad()
def teacher_report(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    targets: Sequence[int],
    *,
    device: str | torch.device = "cpu",
    micro_batch: int = 32,
) -> dict[str, list[float] | float]:
    """Clean (no jitter) teacher CE, target-class probability and top-1 agreement of every image (fp32)."""
    dev = torch.device(device)
    tgt = torch.tensor(list(targets), dtype=torch.long)
    ce, prob, hit = [], [], []
    for s in range(0, images.shape[0], micro_batch):
        logits = logits_fn(images[s : s + micro_batch].to(dev)).float().cpu()
        t = tgt[s : s + micro_batch]
        ce.append(F.cross_entropy(logits, t, reduction="none"))
        prob.append(logits.softmax(-1).gather(1, t[:, None])[:, 0])
        hit.append((logits.argmax(-1) == t).float())
    ce_t, prob_t, hit_t = torch.cat(ce), torch.cat(prob), torch.cat(hit)
    return {
        "ce": ce_t.tolist(),
        "target_prob": prob_t.tolist(),
        "top1_is_target": hit_t.tolist(),
        "mean_ce": float(ce_t.mean()),
        "mean_target_prob": float(prob_t.mean()),
        "top1_rate": float(hit_t.mean()),
    }
