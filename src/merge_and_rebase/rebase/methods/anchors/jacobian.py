"""Jacobian of the task logits w.r.t. the residual stream, in the common coordinates (AGENT_SHARED_TRAINING.md 1.4).

``VisionTail`` runs an OpenCLIP vision tower from the output of block ``start`` (cached by ``prefix``) to the
zero-shot logits ``z = scale * normalize(f) T^T``, adding per-block displacements after chosen blocks. On it:

* ``jvp_logits``: ``J . v`` for displacements ``v`` after several blocks (``torch.func.jvp``, forward mode, MATH
  attention backend);
* ``vjp_logits``: ``g^T J`` (reverse mode, ``torch.autograd.grad``);
* ``common_jvp`` / ``common_vjp``: the same in the common coordinates of a ``ModelFrame``:
  ``J_m(s) = c_m (dz / dh_m(s)) Phi_m^T`` (a displacement ``du`` enters as ``dh = c_m du Phi_m``).

Nothing materialises ``J``; the explicit rows (``jacobian_rows``: one VJP per class) are only used by the
diagnostics on few images. Model computations run in the model dtype (float32); reductions are accumulated in float64
by the callers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from .._shared import _interp_2d_tokens
from .shared_space import ModelFrame

Tensor = torch.Tensor


@contextlib.contextmanager
def math_attention() -> Iterator[None]:
    """Forward-mode AD needs the MATH SDPA kernel on every device (the fused CPU/CUDA kernels have no JVP) and the
    ``nn.MultiheadAttention`` fast path off. (``utils.linearization.forward_ad_safe_attention_context`` pins MATH on
    CUDA only.)"""
    old = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        old = bool(torch.backends.mha.get_fastpath_enabled())
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        if old is not None:
            torch.backends.mha.set_fastpath_enabled(old)


class VisionTail:
    """Logits of an OpenCLIP vision tower as a function of the stream after a block.

    ``text_features``: ``[C, D]`` zero-shot class embeddings (normalised), ``logit_scale``: the classifier's scale.
    Streams are batch-first ``[B, N, d]`` whatever the tower's layout.
    """

    def __init__(self, model, text_features: Tensor, *, logit_scale: float = 100.0, normalize: bool = True) -> None:
        self.visual = model.visual if hasattr(model, "visual") else model
        self.blocks = self.visual.transformer.resblocks
        self.batch_first = bool(getattr(self.visual.transformer, "batch_first", True))
        self.text_features = text_features
        self.logit_scale = float(logit_scale)
        self.normalize = bool(normalize)

    @property
    def depth(self) -> int:
        return len(self.blocks)

    @property
    def device(self) -> torch.device:
        return next(self.visual.parameters()).device

    def _run_block(self, blk, x: Tensor) -> Tensor:
        if self.batch_first:
            return blk(x)
        return blk(x.transpose(0, 1)).transpose(0, 1)

    @torch.no_grad()
    def prefix(self, images: Tensor, start: int) -> Tensor:
        """The stream after block ``start`` (``start = -1``: after ``ln_pre``)."""
        x = self.visual._embeds(images.to(self.device))
        for blk in self.blocks[: start + 1]:
            x = self._run_block(blk, x)
        return x

    @torch.no_grad()
    def streams(self, images: Tensor, blocks: Sequence[int]) -> dict[int, Tensor]:
        """The stream after each of ``blocks`` (one forward)."""
        want = {int(b) for b in blocks}
        x = self.visual._embeds(images.to(self.device))
        out: dict[int, Tensor] = {}
        for j in range(max(want) + 1):
            x = self._run_block(self.blocks[j], x)
            if j in want:
                out[j] = x
        return out

    def features(self, h: Tensor, start: int, deltas: Mapping[int, Tensor] | None = None) -> Tensor:
        deltas = deltas or {}
        bad = [b for b in deltas if b < start or b >= self.depth]
        if bad:
            raise ValueError(f"displacements after blocks {bad} are outside the tail ({start}, {self.depth})")
        x = h + deltas[start] if start in deltas else h
        for j in range(start + 1, self.depth):
            x = self._run_block(self.blocks[j], x)
            if j in deltas:
                x = x + deltas[j]
        pooled, _ = self.visual._pool(x)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        return pooled

    def logits(self, h: Tensor, start: int, deltas: Mapping[int, Tensor] | None = None) -> Tensor:
        f = self.features(h, start, deltas)
        if self.normalize:
            f = F.normalize(f, dim=-1)
        t = self.text_features.to(device=f.device, dtype=f.dtype)
        return self.logit_scale * (f @ t.T)

    @torch.no_grad()
    def full_logits(self, images: Tensor) -> Tensor:
        return self.logits(self.prefix(images, -1), -1)


def jvp_logits(tail: VisionTail, h: Tensor, start: int, tangents: Mapping[int, Tensor]) -> tuple[Tensor, Tensor]:
    """``(z, J . v)``: logits and their directional derivative for displacements ``tangents[block]`` (``[B, N, d]``)
    added after the given blocks."""
    blocks = sorted(int(b) for b in tangents)
    primals = tuple(torch.zeros_like(tangents[b]) for b in blocks)
    tans = tuple(tangents[b] for b in blocks)

    def f(*ds):
        return tail.logits(h, start, dict(zip(blocks, ds, strict=True)))

    with math_attention():
        z, dz = jvp(f, primals, tans)
    return z.detach(), dz.detach()


def vjp_logits(tail: VisionTail, h: Tensor, start: int, blocks: Sequence[int], cotangent: Tensor) -> dict[int, Tensor]:
    """``g^T J`` w.r.t. the displacement after each of ``blocks`` (``[B, N, d]`` each), ``g = cotangent`` ``[B, C]``."""
    blocks = sorted(int(b) for b in blocks)
    ds = [torch.zeros_like(h, requires_grad=True) for _ in blocks]
    with torch.enable_grad():
        z = tail.logits(h, start, dict(zip(blocks, ds, strict=True)))
        grads = torch.autograd.grad(z, ds, grad_outputs=cotangent.to(z.dtype))
    return {b: g.detach() for b, g in zip(blocks, grads, strict=True)}


def _block_tangents(frame: ModelFrame, du: Mapping[int, Tensor]) -> dict[int, Tensor]:
    """Common displacements per grid point -> stream displacements per block (summed if two grid points share one)."""
    out: dict[int, Tensor] = {}
    for j, v in du.items():
        b = frame.blocks[frame.index(j)]
        dh = frame.decode(v, j)
        out[b] = out[b] + dh if b in out else dh
    return out


def common_jvp(tail: VisionTail, frame: ModelFrame, h: Tensor, start: int, du: Mapping[int, Tensor]) -> tuple[Tensor, Tensor]:
    """``(z, J_m . du)`` with ``du[j]`` ``[B, N, r]`` in the common coordinates of grid point ``j``."""
    tangents = {b: t.to(h.dtype) for b, t in _block_tangents(frame, du).items()}
    return jvp_logits(tail, h, start, tangents)


def common_vjp(
    tail: VisionTail, frame: ModelFrame, h: Tensor, start: int, grid: Sequence[int], cotangent: Tensor
) -> dict[int, Tensor]:
    """``g^T J_m`` in the common coordinates: per grid point ``[B, N, r]`` (``c_m (dz/dh) Phi_m^T``)."""
    blocks = {j: frame.blocks[frame.index(j)] for j in grid}
    grads = vjp_logits(tail, h, start, sorted(set(blocks.values())), cotangent)
    return {j: frame.encode_grad(grads[b], j) for j, b in blocks.items()}


def jacobian_rows(
    tail: VisionTail, frame: ModelFrame, h: Tensor, start: int, grid: Sequence[int], *, num_classes: int
) -> dict[int, Tensor]:
    """Explicit ``J_m(s, x)`` in common coordinates: per grid point ``[B, C, N, r]`` (one VJP per class)."""
    b = h.shape[0]
    rows: dict[int, list[Tensor]] = {j: [] for j in grid}
    for c in range(num_classes):
        g = torch.zeros(b, num_classes, device=h.device, dtype=h.dtype)
        g[:, c] = 1.0
        for j, v in common_vjp(tail, frame, h, start, grid, g).items():
            rows[j].append(v)
    return {j: torch.stack(v, dim=1) for j, v in rows.items()}


def to_tokens(v: Tensor, n_tokens: int) -> Tensor:
    """Canonical-grid rows ``[B, N_c, r]`` -> ``n_tokens`` rows (CLS kept, bilinear patch grid; identity if equal)."""
    return _interp_2d_tokens(v, int(n_tokens))


def canonical_adjoint(g: Tensor, canonical_tokens: int) -> Tensor:
    """Adjoint of ``to_tokens`` (``N_c -> N``) applied to a gradient ``[B, ..., N, r]`` over the model's tokens: the
    gradient w.r.t. canonical rows ``[B, ..., N_c, r]``. Exact (autograd through the linear resampling)."""
    n = int(g.shape[-2])
    if n == int(canonical_tokens):
        return g
    lead = g.shape[:-2]
    flat = g.reshape(-1, n, g.shape[-1])
    x = torch.zeros(flat.shape[0], int(canonical_tokens), flat.shape[-1], dtype=flat.dtype, device=flat.device,
                    requires_grad=True)
    with torch.enable_grad():
        y = to_tokens(x, n)
        (gx,) = torch.autograd.grad(y, x, grad_outputs=flat)
    return gx.reshape(*lead, int(canonical_tokens), g.shape[-1])
