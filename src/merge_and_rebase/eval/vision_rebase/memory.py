"""Opt-in GPU-memory reduction for the Ariadne fit with a large source CLIP (``gpu_memory_efficient``).

Without the flag the run keeps both full CLIPs (text towers included) on the GPU and the fit deep-copies the full
source CLIP twice more (source base, source fine-tuned), also on the GPU. With ``gpu_memory_efficient: true``:

* the source classifier is parked on the CPU right after it is built (built on the GPU exactly as before, then moved);
* the two fit copies of the source model are loaded on the CPU as before, then reduced to their visual tower
  (`VisualOnlyModel`); the fit moves a model to the GPU only while it forwards it (capture) or for the whole
  streaming pass, so the source text towers and the idle source CLIP never occupy the GPU;
* source text features (only needed by the balanced calibration mix) are computed with the source CLIP moved to the
  GPU for that call only, so they are produced by the same device kernels as before.

Nothing here changes an arithmetic operation, a dtype or an operand order: whole modules move between devices and the
visual tower that is forwarded is the very same parameter set. Values are bit-identical to the flag-off run on a given
GPU (same kernels, same inputs).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

CONFIG_KEY = "gpu_memory_efficient"


def memory_efficient_enabled(cfg: Mapping[str, Any]) -> bool:
    """True iff the run opted into the low-GPU-memory placement (default off)."""
    return bool(cfg.get(CONFIG_KEY, False))


def release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def park_on_cpu(model: nn.Module) -> None:
    """Move ``model`` to the CPU and return its freed blocks to the driver."""
    model.to("cpu")
    release_cuda_cache()


class VisualOnlyModel(nn.Module):
    """Holds only ``visual`` of an open_clip-like CLIP (the very same module object, no copy).

    The Ariadne capture/fit reads ``model.visual.transformer.resblocks`` and forwards through
    ``_encode_image``, which falls back to ``model.visual(images)`` when there is no ``encode_image``; open_clip's own
    ``encode_image(normalize=False)`` is that same call. No hook, statistic or fit input depends on the text tower.
    """

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual


def visual_only_view(model: nn.Module) -> VisualOnlyModel:
    """Reduce a (fully loaded) CLIP to its visual tower; the text tower is released when ``model`` goes out of scope."""
    view = VisualOnlyModel(model.visual)
    view.train(model.training)
    return view


def source_fit_copy(
    model: nn.Module,
    *,
    memory_efficient: bool,
    loaders: tuple[Callable[[nn.Module], Any], ...] = (),
) -> nn.Module:
    """Deep copy of the source CLIP for one fit endpoint.

    ``loaders`` are callables applied to the full copy (state-dict loads) before it is reduced, so the weights are
    loaded exactly as in the flag-off path.
    """
    copy_ = deepcopy(model)
    for load in loaders:
        load(copy_)
    return visual_only_view(copy_) if memory_efficient else copy_


@contextlib.contextmanager
def temporarily_on(model: nn.Module, device: str | torch.device, *, enabled: bool) -> Iterator[None]:
    """Run a block with ``model`` on ``device`` and put it back where it was (no-op when ``enabled`` is false)."""
    if not enabled:
        yield
        return
    original = next(model.parameters()).device
    model.to(device)
    try:
        yield
    finally:
        model.to(original)
        release_cuda_cache()
