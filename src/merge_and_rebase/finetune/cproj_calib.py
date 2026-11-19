"""Native fine-tuning of ONLY the MLP ``c_proj`` (weight + bias, every visual block) of a target CLIP vision tower.

Rebuttal baseline for Ariadne: same object (the target's ``c_proj`` parameters), frozen zero-shot text head, the
result is a task vector keyed by the target ``state_dict`` names, written in the file format accepted by
``precomputed_tv_dir`` (``{task}_{canonical Ariadne name}_transported_native.pt``) so that the normal
``python -m merge_and_rebase.eval.vision_rebase`` alpha-on-val / test pipeline evaluates it unchanged.

Modes (``--mode``):

* ``budget``          -- the exact Ariadne calibration draw (``paired_calibration``: same task TRAIN loader, seed,
  ``num_batches``, ``batch_size``), cycled over until ``--time-budget-s`` of wall clock is spent (checked after every
  optimizer step, so the budget is overshot by at most one step).
* ``steps``           -- the same calibration draw cycled over for exactly ``--num-steps`` optimizer steps (one per
  calibration batch, constant lr, final weights, val never used); compute-matched baseline, same timed region as
  ``budget``. ``ce`` and ``kd`` only.
* ``converge_calib``  -- the same calibration images, epochs until early stopping on the task val split.
* ``converge_full``   -- the full task train split, classic task-arithmetic recipe: fixed 2000 steps, effective batch
  128 (32 x 4), lr 1e-5, AdamW wd 0.1, cosine with 500 warmup, eval preprocessing (no augmentation, as the repo's
  native trainer), final weights kept (val logged every 500 steps as a diagnostic only).

Objectives (``--objective``): ``ce`` (labelled cross-entropy, default), ``kd`` (logit distillation of the source
fine-tuned model) and ``act`` (activation-displacement distillation: gradient descent, end to end, on
ARIADNE's per-block targets D_j and Q_j (fitted jointly with propagated inputs, no ridge); see the ``act`` section below). ``ce`` and ``kd`` are unchanged by ``act``.

Selection uses the val split only (the carve-out of the TEST split that the Ariadne pipeline uses as its val); the
test split is never touched here. lr is fixed by the task-arithmetic fine-tuning setting (1e-5); ``--lr`` exists
for reproducibility only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore

from ..models.parameter_subsets import resolve_parameter_subset
from ..rebase.registry import canonical_method_name
from ..utils.cost_accounting import PhaseCostRecorder, recording
from ._vision_runtime import ImageEncoder
from .strategies.full import FullFinetune

MODES = ("budget", "converge_calib", "converge_full", "steps")
TRAINABLE_MODES = ("cproj", "full", "lora_cproj", "lora_full")
LORA_MODES = ("lora_cproj", "lora_full")
_CPROJ_BASE_RE = re.compile(r"^visual\.transformer\.resblocks\.\d+\.mlp\.c_proj\.(weight|bias)$")
_LORA_CPROJ_RE = re.compile(r"^visual\.transformer\.resblocks\.\d+\.mlp\.c_proj\.weight$")
_LORA_FULL_RE = re.compile(
    r"^visual\.transformer\.resblocks\.\d+\.(attn\.in_proj_weight|attn\.out_proj\.weight|mlp\.c_fc\.weight|mlp\.c_proj\.weight)$"
)
_ENCODER_PREFIX = "clip_model.model."
_TEACHER_PRETRAINED_DEFAULT = "datacomp_xl_s13b_b90k"
_CONFIG_DIR = Path(__file__).parent / "configs"
_EXCL = "excludes model build / zero-shot head, val evaluation, best-state snapshots and saving"
_TIMED_DEFINITION = {
    "budget": "calibration-data materialization + trainer setup + optimizer steps until the budget is reached "
    f"(wall clock; {_EXCL}); peaks cover the whole region",
    "steps": "calibration-data materialization + trainer setup + the fixed number of optimizer steps "
    f"(wall clock; {_EXCL}); peaks cover the whole region",
    "converge_calib": "calibration-data materialization + trainer setup + summed per-epoch training time "
    f"(wall_seconds = setup + train_seconds; {_EXCL}); peaks = max over setup and per-epoch training brackets",
    "converge_full": "trainer setup + summed training time of the max_steps optimizer steps, timed in val_every-step "
    f"chunks, data loading included (wall_seconds = setup + train_seconds; {_EXCL}); val diagnostics excluded; "
    "peaks = max over setup and per-chunk training brackets",
}


# ---------------------------------------------------------------------------------------------------------------
# trainable subset, init delta, delta extraction
# ---------------------------------------------------------------------------------------------------------------


def _base_name(encoder_param_name: str) -> str:
    if not encoder_param_name.startswith(_ENCODER_PREFIX):
        raise ValueError(f"unexpected parameter name {encoder_param_name!r}")
    return encoder_param_name[len(_ENCODER_PREFIX) :]


def configure_cproj_training(
    encoder: ImageEncoder,
    *,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    scheduler_name: str,
    total_steps: int,
    device: torch.device,
    optimizer: str = "adamw",
    trainable: str = "cproj",
    lora: LoraSet | None = None,
):
    """Freeze everything but the visual ``mlp.c_proj`` weight/bias; return ``(opt, sched, info)`` and assert the set.

    ``trainable="full"``: every float parameter of the image tower (``visual.*``) is trained instead; text tower and
    logit scale stay frozen. ``trainable="lora_cproj"|"lora_full"``: only the LoRA factors of ``lora`` (an
    already-attached :class:`LoraSet`, see ``attach_lora``) are trained; all base weights stay frozen."""
    if trainable not in TRAINABLE_MODES:
        raise ValueError(f"trainable must be one of {TRAINABLE_MODES}, got {trainable!r}")
    for p in encoder.parameters():
        p.requires_grad_(False)
    if trainable in LORA_MODES:
        if lora is None:
            raise ValueError(f"--trainable {trainable} needs an attached LoraSet (attach_lora)")
        lora_params = {id(p) for p in lora.parameters()}
        for p in encoder.parameters():
            if id(p) in lora_params:
                p.requires_grad_(True)
        opt, sched, info = FullFinetune().configure(
            model=encoder,
            lr=lr,
            weight_decay=weight_decay,
            warmup_length=warmup_steps,
            scheduler_name=scheduler_name,
            optimizer=optimizer,
            steps=max(1, int(total_steps)),
            device=device,
            strategy_cfg={"params": {"trainable_params": "all_trainable"}},  # keeps the requires_grad set above
        )
        trained = [p for p in encoder.parameters() if p.requires_grad]
        assert trained and {id(p) for p in trained} == lora_params, "trainable set differs from the LoRA factors"
        info = dict(info)
        info.update(
            {
                "trainable_tensors": len(trained),
                "trainable_mode": trainable,
                "lora_rank": lora.rank,
                "lora_alpha": lora.alpha,
                "lora_scale": lora.scale,
                "lora_targets": len(lora.names),
                "lora_trainable_params": int(sum(p.numel() for p in trained)),
            }
        )
        return opt, sched, info
    if trainable == "full":
        for name, p in encoder.named_parameters():
            if _base_name(name).startswith("visual.") and p.is_floating_point():
                p.requires_grad_(True)
        opt, sched, info = FullFinetune().configure(
            model=encoder,
            lr=lr,
            weight_decay=weight_decay,
            warmup_length=warmup_steps,
            scheduler_name=scheduler_name,
            optimizer=optimizer,
            steps=max(1, int(total_steps)),
            device=device,
            strategy_cfg={"params": {"trainable_params": "all_trainable"}},  # keeps the requires_grad set above
        )
        names = sorted(n for n, p in encoder.named_parameters() if p.requires_grad)
        assert names and all(_base_name(n).startswith("visual.") for n in names), "non-visual parameter is trainable"
        info = dict(info)
        info.update({"trainable_tensors": len(names), "trainable_mode": "full"})
        return opt, sched, info
    for name, p in encoder.named_parameters():
        if _base_name(name).startswith("visual.") and _CPROJ_BASE_RE.fullmatch(_base_name(name)):
            p.requires_grad_(True)
    opt, sched, info = FullFinetune().configure(
        model=encoder,
        lr=lr,
        weight_decay=weight_decay,
        warmup_length=warmup_steps,
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        steps=max(1, int(total_steps)),
        device=device,
        strategy_cfg={"params": {"trainable_params": "c_proj_only"}},
    )
    trainable = sorted(n for n, p in encoder.named_parameters() if p.requires_grad)
    expected = sorted(resolve_parameter_subset(encoder, "c_proj_only").parameter_names)
    assert trainable == expected and trainable, "trainable set differs from the c_proj_only subset"
    assert all(_CPROJ_BASE_RE.fullmatch(_base_name(n)) for n in trainable), "non-c_proj parameter is trainable"
    n_blocks = len({re.search(r"resblocks\.(\d+)\.", n).group(1) for n in trainable})  # type: ignore[union-attr]
    assert len(trainable) == 2 * n_blocks, "expected one c_proj weight and one bias per block"
    info = dict(info)
    info.update({"trainable_tensors": len(trainable), "trainable_blocks": n_blocks, "trainable_mode": "cproj"})
    return opt, sched, info


class _LoraParam(torch.nn.Module):
    """Parametrization ``W -> W + scale * B @ A`` (``A``: r x in, kaiming-uniform(a=sqrt(5)) as PEFT; ``B``: out x r, zeros,
    so the effective weight equals the base weight at initialisation)."""

    def __init__(self, shape: Sequence[int], rank: int, scale: float, gen: torch.Generator, device, dtype) -> None:
        super().__init__()
        out_f, in_f = int(shape[0]), int(shape[1])
        a = torch.empty(rank, in_f, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
        self.A = torch.nn.Parameter(a.to(device=device, dtype=dtype))
        self.B = torch.nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        self.scale = float(scale)

    def delta(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + self.delta()


class LoraSet:
    """The LoRA parametrizations attached to a model: ``names`` (target ``state_dict`` names of the adapted weights),
    ``modules`` (one :class:`_LoraParam` each, same order)."""

    def __init__(self, model: torch.nn.Module, names: list[str], modules: list[_LoraParam], rank: int, alpha: float):
        self.model, self.names, self.modules = model, names, modules
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.module_list = torch.nn.ModuleList(modules)  # for state_dict snapshot / restore of the best epoch

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.module_list.parameters())

    def deltas(self) -> dict[str, torch.Tensor]:
        """Merged update ``scale * B @ A`` of every adapted weight (fp32, cpu), keyed by target state_dict name."""
        with torch.no_grad():
            return {n: m.delta().detach().to("cpu", torch.float32).clone() for n, m in zip(self.names, self.modules, strict=True)}

    def detach(self) -> None:
        """Remove the parametrizations, restoring the untouched base weights (``leave_parametrized=False``)."""
        from torch.nn.utils import parametrize

        for n in self.names:
            prefix, _, leaf = n.rpartition(".")
            parametrize.remove_parametrizations(self.model.get_submodule(prefix), leaf, leave_parametrized=False)


def lora_target_names(model: torch.nn.Module, trainable: str) -> list[str]:
    """Target ``state_dict`` names adapted by ``lora_cproj`` (every visual ``mlp.c_proj.weight``) or ``lora_full``
    (``attn.in_proj_weight``, ``attn.out_proj.weight``, ``mlp.c_fc.weight``, ``mlp.c_proj.weight`` of every visual block)."""
    if trainable not in LORA_MODES:
        raise ValueError(f"trainable must be one of {LORA_MODES}, got {trainable!r}")
    pat = _LORA_CPROJ_RE if trainable == "lora_cproj" else _LORA_FULL_RE
    return [k for k in model.state_dict() if pat.fullmatch(k)]


def attach_lora(
    model: torch.nn.Module,
    trainable: str,
    *,
    rank: int = 16,
    alpha: float | None = None,
    seed: int = 0,
) -> LoraSet:
    """Register ``W_eff = W + (alpha/r) B A`` on every targeted weight (``torch.nn.utils.parametrize``).

    The base weight is frozen and left untouched (it lives in ``parametrizations.<leaf>.original``); ``B = 0`` so the
    model starts exactly at the base. ``A`` is seeded with its own CPU generator (independent of the global RNG)."""
    from torch.nn.utils import parametrize

    if rank < 1:
        raise ValueError("--lora-rank must be >= 1")
    alpha = float(rank) if alpha is None else float(alpha)
    if not alpha > 0:
        raise ValueError("--lora-alpha must be > 0")
    names = lora_target_names(model, trainable)
    if not names:
        raise ValueError(f"no weight matched for --trainable {trainable}")
    if trainable == "lora_cproj":
        assert all(".mlp.c_proj.weight" in n for n in names)
    sd = model.state_dict()
    gen = torch.Generator().manual_seed(int(seed))
    mods = []
    for n in names:
        w = sd[n]
        if w.ndim != 2:
            raise ValueError(f"LoRA target {n} is not a 2-D matrix: {tuple(w.shape)}")
        m = _LoraParam(w.shape, rank, alpha / rank, gen, w.device, w.dtype)
        prefix, _, leaf = n.rpartition(".")
        parametrize.register_parametrization(model.get_submodule(prefix), leaf, m)
        mods.append(m)
    return LoraSet(model, names, mods, rank, alpha)


def lora_delta_dict(lora: LoraSet, base: dict[str, torch.Tensor], keys: Sequence[str]) -> dict[str, torch.Tensor]:
    """Task vector in the ``cproj`` / ``full`` file format: ``scale * B @ A`` on the adapted weights, zeros elsewhere."""
    d = lora.deltas()
    return {k: d[k] if k in d else torch.zeros_like(base[k]) for k in keys}


def cproj_keys(model: torch.nn.Module) -> list[str]:
    """Target ``state_dict`` names of the c_proj weights/biases (``model`` = the open_clip model)."""
    return [k for k in model.state_dict() if _CPROJ_BASE_RE.fullmatch(k)]


def visual_float_keys(model: torch.nn.Module) -> list[str]:
    """Target ``state_dict`` names of every float ``visual.*`` tensor (``trainable="full"``; ``model`` = open_clip model)."""
    return [k for k, v in model.state_dict().items() if k.startswith("visual.") and v.is_floating_point()]


def micro_split(batches: Sequence[Any], micro_batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split every ``(x, target)`` batch (``target`` = labels or cached teacher logits) into consecutive micro-batches of
    ``micro_batch`` rows. Used with ``accumulate = batch / micro_batch`` so one optimizer step is still one calibration
    batch and (batch-mean losses, each micro-batch weighted ``1/accumulate`` by ``run_epoch``) the gradient equals the
    full-batch one."""
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in batches:
        n = x.shape[0]
        if n % micro_batch:
            raise ValueError(f"--micro-batch {micro_batch} must divide the calibration batch size {n}")
        for s0 in range(0, n, micro_batch):
            out.append((x[s0 : s0 + micro_batch], y[s0 : s0 + micro_batch]))
    return out


def snapshot(model: torch.nn.Module, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: sd[k].detach().to("cpu", torch.float32).clone() for k in keys}


@torch.no_grad()
def apply_init_delta(model: torch.nn.Module, delta: dict[str, torch.Tensor], alpha: float) -> list[str]:
    """``params += alpha * delta`` for every key of ``delta`` (target ``state_dict`` names, shapes checked)."""
    sd = model.state_dict()
    for k, v in delta.items():
        if k not in sd or tuple(sd[k].shape) != tuple(v.shape):
            raise ValueError(f"init delta key {k!r} missing from the target or shape mismatch")
    for k, v in delta.items():
        sd[k].add_(v.to(sd[k].device, sd[k].dtype), alpha=float(alpha))
    return list(delta)


def extract_delta(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    keys: Sequence[str],
) -> dict[str, torch.Tensor]:
    """``(current - base)`` for ``keys``, fp32 cpu, keyed by target state_dict names."""
    cur = snapshot(model, keys)
    return {k: cur[k] - base[k] for k in keys}


# ---------------------------------------------------------------------------------------------------------------
# training loops
# ---------------------------------------------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(
    encoder: ImageEncoder,
    batches: Sequence[Any],
    opt: torch.optim.Optimizer,
    sched: Callable[[int], None],
    *,
    accumulate: int,
    grad_clip: float,
    device: torch.device,
    global_step: int,
    losses: list[float],
    after_step: Callable[[int], bool] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, bool]:
    """One pass over ``batches`` (accumulating ``accumulate`` batches per optimizer step).

    ``after_step(global_step)`` runs after every optimizer step; returning True stops the pass immediately.
    Returns ``(global_step, stopped)``. Per-step losses (mean over the accumulation window) are appended to ``losses``.
    ``loss_fn(student_logits, target)`` defaults to cross-entropy (``target`` = labels); for ``--objective kd`` the
    batches are ``(x, teacher_logits)`` and ``loss_fn`` is ``kd_loss`` (no label tensor exists in the batches).
    """
    encoder.train()
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    n = len(batches)
    window = 0
    win_size = 1
    win_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(batches):
        if window == 0:
            win_size = min(accumulate, n - i)
            win_loss = 0.0
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = loss_fn(encoder(x), y)
        (loss / win_size).backward()
        win_loss += float(loss.item()) / win_size
        window += 1
        if window == win_size:
            if grad_clip > 0:
                for g in opt.param_groups:  # per group: Q (act learned) must not rescale the c_proj step
                    params = [p for p in g["params"] if p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
            sched(global_step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            window = 0
            losses.append(win_loss)
            if after_step is not None and after_step(global_step):
                return global_step, True
    return global_step, False


def train_budget(
    encoder: ImageEncoder,
    batches: Sequence[Any],
    opt: torch.optim.Optimizer,
    sched: Callable[[int], None],
    *,
    time_budget_s: float,
    t0: float,
    grad_clip: float,
    device: torch.device,
    clock: Callable[[], float] = time.perf_counter,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    accumulate: int = 1,
) -> dict[str, Any]:
    """Cycle over the fixed ``batches`` until ``clock() - t0 >= time_budget_s`` (checked after each optimizer step).

    ``t0`` is the start of the timed region (so data materialization done before this call is charged to the budget).
    At least one step is always taken.
    """
    losses: list[float] = []
    step_end: list[float] = []

    def after_step(_step: int) -> bool:
        _sync(device)
        now = clock() - t0
        step_end.append(now)
        return now >= time_budget_s

    step, epochs_done, stopped = 0, 0, False
    while not stopped:
        step, stopped = run_epoch(
            encoder,
            batches,
            opt,
            sched,
            accumulate=accumulate,
            grad_clip=grad_clip,
            device=device,
            global_step=step,
            losses=losses,
            after_step=after_step,
            loss_fn=loss_fn,
        )
        if not stopped:
            epochs_done += 1
    return {
        "steps": step,
        "epochs": step * accumulate / max(1, len(batches)),
        "train_seconds": step_end[-1],
        "step_end_seconds": step_end,
        "train_loss_steps": losses,
    }


def train_fixed_steps(
    encoder: ImageEncoder,
    batches: Sequence[Any],
    opt: torch.optim.Optimizer,
    sched: Callable[[int], None],
    *,
    num_steps: int,
    grad_clip: float,
    device: torch.device,
    clock: Callable[[], float] = time.perf_counter,
    t0: float | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    accumulate: int = 1,
) -> dict[str, Any]:
    """Cycle over the fixed ``batches`` for exactly ``num_steps`` optimizer steps (``accumulate`` batches each)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    t0 = clock() if t0 is None else t0
    losses: list[float] = []
    step_end: list[float] = []

    def after_step(step: int) -> bool:
        _sync(device)
        step_end.append(clock() - t0)
        return step >= num_steps

    step, stopped = 0, False
    while not stopped:
        step, stopped = run_epoch(
            encoder, batches, opt, sched, accumulate=accumulate, grad_clip=grad_clip, device=device,
            global_step=step, losses=losses, after_step=after_step, loss_fn=loss_fn,
        )
    return {
        "steps": step,
        "epochs": step * accumulate / max(1, len(batches)),
        "train_seconds": step_end[-1],
        "step_end_seconds": step_end,
        "train_loss_steps": losses,
    }


def train_converge(
    encoder: ImageEncoder,
    train_batches: Callable[[], Sequence[Any]],
    val_loader: Any,
    opt: torch.optim.Optimizer,
    sched: Callable[[int], None],
    *,
    keys: Sequence[str],
    max_epochs: int,
    patience: int,
    accumulate: int,
    grad_clip: float,
    device: torch.device,
    start_val: float | None = None,
    selection: str = "best_val",
    recorder: PhaseCostRecorder | None = None,
    log: Callable[[str], None] = print,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    extra_module: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Epochs with val early stopping (val only); restores the best-val weights into ``encoder`` at the end.

    ``selection="final"`` (classic task-arithmetic recipe): all ``max_epochs`` epochs are run, no early stopping and
    no best-state restore (the final weights are kept; val is evaluated and logged per epoch as a diagnostic only,
    outside the timed region); ``best_epoch`` is None and ``val_acc_final`` is returned.

    Epoch 0 (the weights on entry: zero-shot, or the ``--init-delta`` start) is a candidate: ``best_val`` and the
    best state are seeded with it (``start_val`` if given, else evaluated here), and ``best_epoch == 0`` when no
    trained epoch strictly improves on it. ``train_batches()`` returns the (re-iterable) batches of one epoch (a
    fixed list or a shuffling DataLoader). Val evaluation and best-state snapshots are NOT part of
    ``train_seconds`` nor of the peaks (``train_cuda_peak_bytes`` / ``train_host_peak_bytes`` are the max over
    per-epoch training brackets of ``recorder``; 0 without a recorder).
    """
    model = encoder.clip_model.model
    t = time.perf_counter()
    best_val = float(start_val) if start_val is not None else float(encoder.top1(val_loader, str(device)))
    eval_seconds = 0.0 if start_val is not None else time.perf_counter() - t
    if selection not in ("final", "best_val"):
        raise ValueError(f"selection must be 'final' or 'best_val', got {selection!r}")
    best_epoch, best_state, bad, step = 0, (snapshot(model, keys) if selection != "final" else None), 0, 0

    def _extra_snap():  # e.g. the learned Q_j of --objective act: restored together with the best-epoch weights
        return None if extra_module is None else {k: v.detach().clone() for k, v in extra_module.state_dict().items()}

    best_extra = _extra_snap() if selection != "final" else None
    losses: list[float] = []
    epoch_log: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "val_acc": best_val}]
    train_seconds = 0.0
    cuda_peak = host_peak = 0.0
    for epoch in range(1, max_epochs + 1):
        mark = recorder.mark() if recorder is not None else None
        _sync(device)
        t = time.perf_counter()
        n0 = len(losses)
        step, _ = run_epoch(
            encoder,
            train_batches(),
            opt,
            sched,
            accumulate=accumulate,
            grad_clip=grad_clip,
            device=device,
            global_step=step,
            losses=losses,
            loss_fn=loss_fn,
        )
        _sync(device)
        train_seconds += time.perf_counter() - t
        if recorder is not None:
            c, h = recorder.peaks_since(mark)
            cuda_peak, host_peak = max(cuda_peak, c), max(host_peak, h)
        t = time.perf_counter()
        val = float(encoder.top1(val_loader, str(device)))
        mean_loss = sum(losses[n0:]) / max(1, len(losses) - n0)
        epoch_log.append({"epoch": epoch, "train_loss": mean_loss, "val_acc": val})
        log(f"epoch {epoch:03d}/{max_epochs} loss={mean_loss:.4f} val={val:.4f} best={best_val:.4f}")
        stop = False
        if selection == "final":
            pass
        elif val > best_val:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = snapshot(model, keys)
            best_extra = _extra_snap()
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping at epoch {epoch} (patience {patience})")
                stop = True
        eval_seconds += time.perf_counter() - t
        if stop:
            break
    val_final = epoch_log[-1]["val_acc"]
    if selection != "final":
        sd = model.state_dict()
        with torch.no_grad():
            for k, v in best_state.items():
                sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
        if extra_module is not None and best_extra is not None:
            extra_module.load_state_dict(best_extra)
    return {
        "selection": selection,
        "val_acc_final": val_final,
        "steps": step,
        "epochs": len(epoch_log) - 1,
        "best_epoch": best_epoch if selection != "final" else None,
        "best_val_acc": best_val if selection != "final" else None,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "train_cuda_peak_bytes": cuda_peak,
        "train_host_peak_bytes": host_peak,
        "epoch_log": epoch_log,
        "train_loss_steps": losses,
    }


def train_steps(
    encoder: ImageEncoder,
    loader: Any,
    val_loader: Any,
    opt: torch.optim.Optimizer,
    sched: Callable[[int], None],
    *,
    max_steps: int,
    eval_every: int,
    accumulate: int,
    grad_clip: float,
    device: torch.device,
    start_val: float | None = None,
    recorder: PhaseCostRecorder | None = None,
    log: Callable[[str], None] = print,
    teacher: ImageEncoder | None = None,
    kd_temperature: float = 2.0,
) -> dict[str, Any]:
    """Fixed-step training (classic task-arithmetic recipe): exactly ``max_steps`` optimizer steps, final weights kept.

    The (shuffling) ``loader`` is iterated repeatedly, re-created (hence reshuffled) every pass. Every optimizer step
    accumulates exactly ``accumulate`` batches, windows may span a pass boundary, so there is never a partial-window
    step. Val is evaluated every ``eval_every`` steps and at the end (diagnostic only; outside ``train_seconds`` and
    the peaks). ``epochs`` is the number of passes over the loader (float).

    ``teacher`` (kd objective): the frozen teacher's logits on the SAME batch ``x`` are computed on the fly inside the
    timed chunks (no cache) and the loss is ``kd_loss``; the label ``y`` is then never moved to the device nor used.
    """
    if max_steps < 1 or eval_every < 1:
        raise ValueError("max_steps and eval_every must be >= 1")
    loss_fn = torch.nn.CrossEntropyLoss()
    if teacher is not None:
        teacher.eval()
    t = time.perf_counter()
    v0 = float(start_val) if start_val is not None else float(encoder.top1(val_loader, str(device)))
    eval_seconds = 0.0 if start_val is not None else time.perf_counter() - t
    val_log: list[dict[str, Any]] = [{"step": 0, "val_acc": v0}]
    it = iter(loader)
    step = batches_used = 0
    losses: list[float] = []
    train_seconds = 0.0
    cuda_peak = host_peak = 0.0
    while step < max_steps:
        chunk = min(eval_every, max_steps - step)
        mark = recorder.mark() if recorder is not None else None
        _sync(device)
        t = time.perf_counter()
        encoder.train()
        for _ in range(chunk):
            opt.zero_grad(set_to_none=True)
            win_loss = 0.0
            for _ in range(accumulate):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)
                if teacher is None:
                    loss = loss_fn(encoder(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
                else:
                    xd = x.to(device, non_blocking=True)
                    with torch.no_grad():
                        t_logits = teacher(xd)
                    loss = kd_loss(encoder(xd), t_logits, kd_temperature)
                (loss / accumulate).backward()
                win_loss += float(loss.item()) / accumulate
                batches_used += 1
            if grad_clip > 0:
                for g in opt.param_groups:  # per group: Q (act learned) must not rescale the c_proj step
                    params = [p for p in g["params"] if p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
            sched(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(win_loss)
        _sync(device)
        train_seconds += time.perf_counter() - t
        if recorder is not None:
            c, h = recorder.peaks_since(mark)
            cuda_peak, host_peak = max(cuda_peak, c), max(host_peak, h)
        t = time.perf_counter()
        val = float(encoder.top1(val_loader, str(device)))
        eval_seconds += time.perf_counter() - t
        val_log.append({"step": step, "val_acc": val})
        log(f"step {step}/{max_steps} loss={losses[-1]:.4f} val={val:.4f}")
    return {
        "selection": "final",
        "val_acc_final": val_log[-1]["val_acc"],
        "steps": step,
        "epochs": batches_used / max(1, len(loader)),
        "batches_used": batches_used,
        "best_epoch": None,
        "best_val_acc": None,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "train_cuda_peak_bytes": cuda_peak,
        "train_host_peak_bytes": host_peak,
        "val_log": val_log,
        "epoch_log": None,
        "train_loss_steps": losses,
    }


# ---------------------------------------------------------------------------------------------------------------
# label-free knowledge distillation (rebuttal baseline: source fine-tuned model -> target base, calibration set)
# ---------------------------------------------------------------------------------------------------------------

_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """``T^2 * KL(softmax(teacher/T) || softmax(student/T))``, mean over the batch. No labels."""
    t = float(temperature)
    log_p_s = torch.log_softmax(student_logits.float() / t, dim=-1)
    p_t = torch.softmax(teacher_logits.detach().float() / t, dim=-1)
    return (t * t) * torch.nn.functional.kl_div(log_p_s, p_t, reduction="batchmean")


@torch.no_grad()
def compute_teacher_logits(teacher: ImageEncoder, batches: Sequence[Any], device: torch.device) -> list[torch.Tensor]:
    """Frozen teacher logits (eval, no grad) of every batch's images, computed once; labels are not touched."""
    teacher.eval()
    return [teacher(b[0].to(device, non_blocking=True)).detach() for b in batches]


def kd_batches(batches: Sequence[Any], teacher_logits: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``(x, teacher_logits)`` pairs: the label tensor of the calibration batches is dropped here, by construction."""
    if len(batches) != len(teacher_logits):
        raise ValueError("one teacher-logit tensor per calibration batch is required")
    return [(b[0], t) for b, t in zip(batches, teacher_logits, strict=True)]


def load_teacher_visual(model: torch.nn.Module, ft: dict[str, torch.Tensor]) -> int:
    """Copy every float ``visual.*`` tensor of the fine-tuned state dict ``ft`` into ``model`` (fail on missing/shape)."""
    sd = model.state_dict()
    vis = [k for k, v in sd.items() if k.startswith("visual.") and v.is_floating_point()]
    missing = [k for k in vis if k not in ft]
    if missing:
        raise ValueError(f"{len(missing)} visual keys missing in the teacher checkpoint, e.g. {missing[:3]}")
    bad = [k for k in vis if tuple(ft[k].shape) != tuple(sd[k].shape)]
    if bad:
        raise ValueError(f"shape mismatch in the teacher checkpoint for {bad[:3]}")
    with torch.no_grad():
        for k in vis:
            sd[k].copy_(ft[k].detach().to(sd[k].device, sd[k].dtype))
    return len(vis)


def preprocess_signature(preprocess: Any) -> str:
    """``repr`` of a preprocessing transform without object addresses (compared to decide view reuse)."""
    return _ADDR_RE.sub("", repr(preprocess))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_objective_args(args: argparse.Namespace) -> None:
    if getattr(args, "trainable", "cproj") in LORA_MODES:
        if args.objective == "act":
            raise ValueError("--objective act is not supported with LoRA (--trainable lora_*): use ce or kd")
        if getattr(args, "init_delta", None):
            raise ValueError("--init-delta is not supported with LoRA (--trainable lora_*)")
    teacher_given = [args.teacher_model, args.teacher_ckpt]
    act_flags = (
        args.act_cache_dtype != "fp32"
        or args.act_cache_max_gb != 48.0
        or args.act_micro_batch != 0
        or args.act_q != "procrustes"
        or args.act_align_weight != 0.0
        or args.act_q_lr is not None
        or args.act_q_init is not None
    )
    if args.objective in ("kd", "act"):
        if not all(teacher_given):
            raise ValueError(f"--objective {args.objective} requires --teacher-model and --teacher-ckpt")
    if args.objective == "kd":
        if not args.kd_temperature > 0:
            raise ValueError("--kd-temperature must be > 0")
    elif args.objective == "act":
        if args.kd_temperature != 2.0:
            raise ValueError("--kd-temperature is only valid with --objective kd")
    elif any(teacher_given) or args.teacher_pretrained != _TEACHER_PRETRAINED_DEFAULT or args.kd_temperature != 2.0:
        raise ValueError("--teacher-* / --kd-temperature are only valid with --objective kd")
    if args.objective != "act" and act_flags:
        raise ValueError("--act-* options are only valid with --objective act")
    if args.objective == "act" and args.act_align_weight < 0:
        raise ValueError("--act-align-weight must be >= 0")
    if args.objective == "act" and args.act_q == "procrustes" and (
        args.act_align_weight != 0.0 or args.act_q_lr is not None or args.act_q_init is not None
    ):
        raise ValueError("--act-align-weight / --act-q-lr / --act-q-init only apply with --act-q learned")


def _task_classnames(args: argparse.Namespace, loaders: Any) -> list[str]:
    from ..eval.utils import humanize

    return [humanize(c) for c in loaders.classnames] if args.humanize else list(loaders.classnames)


def _build_vision_loaders_for(args: argparse.Namespace, preprocess: Any, batch_size: int, split_seed: int):
    from ..data.vision_loaders import build_vision_loaders, load_hf_splits
    from ..eval.datasets.vision8_14_20 import SUITES

    hf_path, hf_config, split_map = SUITES[args.suite].resolver(args.task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    return build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        val_fraction=args.val_fraction,
        seed=split_seed,
    )


def _build_teacher(args: argparse.Namespace, target_clf: Any, loaders: Any, batch_size: int, split_seed: int):
    """Frozen teacher (source open_clip model + fine-tuned visual weights + its own zero-shot head), untimed.

    Returns ``(teacher_encoder, teacher_loaders_or_None, view_mode)``: ``None`` loaders / ``"shared_target_view"`` when
    the teacher preprocessing equals the target one (target-view tensors are reused), else a separately preprocessed
    loader set of the same splits (same seeds) and ``"separate_teacher_view"``.
    """
    from ..data.templates import get_templates
    from ..io.ckpt import load_ckpt
    from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

    device = torch.device(args.device)
    cfg = OpenClipBuildConfig(
        model_name=args.teacher_model, pretrained=args.teacher_pretrained, device=args.device, dtype="fp32"
    )
    clf = OpenClipClassifier.build(cfg)
    n_loaded = load_teacher_visual(clf.model, load_ckpt(args.teacher_ckpt))
    task_cfg = OpenClipBuildConfig(
        model_name=args.teacher_model,
        pretrained=args.teacher_pretrained,
        device=args.device,
        dtype="fp32",
        prompt_templates=get_templates(args.task),
    )
    clf.build_zeroshot_text_features(_task_classnames(args, loaders), task_cfg)
    encoder = ImageEncoder(clf).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"teacher: {args.teacher_model} with {n_loaded} visual tensors from {args.teacher_ckpt}")
    same = preprocess_signature(clf.preprocess) == preprocess_signature(target_clf.preprocess)
    if same:
        return encoder, None, "shared_target_view"
    return encoder, _build_vision_loaders_for(args, clf.preprocess, batch_size, split_seed), "separate_teacher_view"


# ---------------------------------------------------------------------------------------------------------------
# activation-displacement distillation (--objective act): gradient descent on ARIADNE's per-block targets
# ---------------------------------------------------------------------------------------------------------------
#
# Definitions (docs/.local/Ariadne_final.pdf Sec. 3.1, Eq. 1-3). Relative-depth map pi(j) = round(j (Ds-1)/(Dt-1)).
# S_base / S_ft: residual stream after source block pi(j) of the source base / source FT; T_j^base: residual stream
# after target block j; source tokens are interpolated to the target grid (ARIADNE's ``_aligned``). Q_j = U V^T from the
# thin SVD of the centered cross-covariance S~_base^T T~_base (ARIADNE's ``prepare_direct_residual_streaming``, called
# here verbatim, same calibration draw). D_j = (S_ft - S_base) Q_j, UNCENTERED: ARIADNE's ridge fits
# ``H dW + 1 beta^T ~ D_j`` with a free bias beta_j, which equals the centered target with a free bias; here the
# trainable ``c_proj.bias`` is that free bias, so the uncentered D_j is the target for which the per-block minimiser
# coincides with ARIADNE's ridge objective without the ridge penalty, for fixed inputs. With propagated inputs the
# joint optimum corresponds to ARIADNE's cumulative/sequential target, not the independent per-block fit (whose
# double counting the alpha search shrinks). No ridge: only early stopping / finite steps / AdamW weight decay.
# Loss: L_disp = mean_j mean_{rows,dims} || T_j(student) - T_j(base) - D_j ||^2 (uniform row weights, CLS included, as
# ARIADNE's default ``alignment_row_weighting='uniform'``; blocks weighted equally). Real propagated inputs: the
# student's residual stream after block j with all c_proj updates active.
# Cache (CPU, per calibration batch, stacked over blocks): dS = S_ft - S_base (aligned), S_base (aligned), T_base. D_j
# is recomputed as dS Q_j at every step (cheap), so the same cache serves fixed and learned Q_j. Default fp32 (a
# fp16 T_base would put rounding noise of order 1e-3 |T| next to displacements much smaller than |T|).
# --act-q learned: Q_j trainable on the rectangular Stiefel manifold (torch orthogonal parametrisation, init = the
# Procrustes solution) with the alignment term of Eq. 1 added (see ActObjective).


class ActTargets:
    """Cached calibration quantities of the act objective (all CPU) and ARIADNE's Q_j with its check statistics."""

    def __init__(self) -> None:
        self.pairing: tuple[int, ...] = ()
        self.source_depth = 0
        self.q0: list[torch.Tensor] = []  # ARIADNE's Procrustes Q_j (float32, [ds, dt])
        self.mu_s: torch.Tensor | None = None  # [J, ds] mean of the aligned S_base bank (ARIADNE's)
        self.mu_t: torch.Tensor | None = None  # [J, dt]
        self.d_s: list[torch.Tensor] = []  # per batch [J, B, T, ds]  S_ft - S_base (aligned)
        self.s_base: list[torch.Tensor] = []  # per batch [J, B, T, ds]
        self.t_base: list[torch.Tensor] = []  # per batch [J, B, T, dt]
        self.tc_norm2: list[float] = []  # per block ||T_base - mu_t||_F^2 over the whole bank
        self.rows_total = 0
        self.target_batches: list[Any] = []
        self.calibration: dict[str, Any] = {}
        self.q_info: list[dict[str, Any]] = []
        self.cache_info: dict[str, Any] = {}


def build_act_targets(
    target_model: torch.nn.Module,
    source_base_model: torch.nn.Module,
    source_ft_model: torch.nn.Module,
    source_loader: Any,
    target_loader: Any,
    *,
    num_batches: int,
    seed: int,
    device: torch.device,
    cache_dtype: torch.dtype = torch.float32,
    max_cache_gb: float = 48.0,
) -> ActTargets:
    """Q_j through ARIADNE's own ``prepare_direct_residual_streaming`` (default alignment: polar, uniform rows), then one
    lockstep sweep (``iter_capture_tokens``, as ARIADNE's sequential fit) caching dS / S_base / T_base per batch.

    ``target_model`` must hold the base weights. Also records, per block, ARIADNE's Procrustes quality: residual
    ``||(S~ Q - T~)||_F / ||T~||_F`` on the calibration bank, ``||Q||_F``, the cross-covariance rank and a sha256 of Q.
    """
    from ..rebase.discrete_layer_match import DiscreteLayerPairing
    from ..rebase.methods._ariadne.capture import iter_capture_tokens
    from ..rebase.methods._ariadne.layouts import _aligned, _layout_for
    from ..rebase.methods._ariadne.streaming import prepare_direct_residual_streaming

    layout = _layout_for(None)
    pairing = DiscreteLayerPairing.compute(layout.block_count(source_base_model), layout.block_count(target_model))
    n_blocks = pairing.target_depth
    prepared = prepare_direct_residual_streaming(
        source_base_model,
        target_model,
        source_loader,
        target_loader,
        pairing,
        num_batches=num_batches,
        seed=seed,
        device=device,
        source_ft_model=source_ft_model,
    )
    out = ActTargets()
    out.pairing = tuple(pairing.pairing)
    out.source_depth = pairing.source_depth
    out.q0 = [prepared["q_by_position"][j].float().cpu() for j in range(n_blocks)]
    out.mu_s = torch.stack([prepared["source_mean_by_position"][j].float() for j in range(n_blocks)])
    out.mu_t = torch.stack([prepared["target_mean_by_position"][j].float() for j in range(n_blocks)])
    out.target_batches = prepared["target_batches"]
    out.calibration = prepared["calibration"]
    q_dev = [q.to(device) for q in out.q0]
    mu_s = out.mu_s.to(device)
    mu_t = out.mu_t.to(device)
    src_req = {str(i): (i, "boundary") for i in sorted(set(pairing.pairing))}
    tgt_req = {str(j): (j, "boundary") for j in range(n_blocks)}
    res2 = [0.0] * n_blocks
    tc2 = [0.0] * n_blocks
    gens = (
        iter_capture_tokens(source_base_model, prepared["source_batches"], src_req, device, store_device=device),
        iter_capture_tokens(source_ft_model, prepared["source_batches"], src_req, device, store_device=device),
        iter_capture_tokens(target_model, out.target_batches, tgt_req, device, store_device=device),
    )
    import contextlib

    n_rows = 0
    with contextlib.ExitStack() as stack:
        for gen in gens:
            stack.enter_context(contextlib.closing(gen))
        for sb, sf, tg in zip(*gens, strict=True):
            slabs: list[torch.Tensor] | None = None
            for j in range(n_blocks):
                i = pairing.pairing[j]
                t = tg[str(j)]
                b = _aligned([sb[str(i)]], [t])[0]
                f = _aligned([sf[str(i)]], [t])[0]
                if slabs is None:
                    shapes = (
                        (n_blocks, *b.shape),
                        (n_blocks, *b.shape),
                        (n_blocks, *t.shape),
                    )
                    per_batch = sum(math.prod(sh) for sh in shapes) * torch.empty((), dtype=cache_dtype).element_size()
                    if per_batch * len(out.target_batches) > max_cache_gb * 1e9:
                        raise RuntimeError(
                            f"act cache would need {per_batch * len(out.target_batches) / 1e9:.1f} GB host RAM "
                            f"(> --act-cache-max-gb {max_cache_gb}); lower --num-batches/--batch-size or raise the limit"
                        )
                    slabs = [torch.empty(sh, dtype=cache_dtype) for sh in shapes]
                slabs[0][j] = (f - b).to(cache_dtype).cpu()
                slabs[1][j] = b.to(cache_dtype).cpu()
                slabs[2][j] = t.to(cache_dtype).cpu()
                tc = t - mu_t[j]
                r = (b - mu_s[j]) @ q_dev[j] - tc
                res2[j] += float((r.double() ** 2).sum().item())
                tc2[j] += float((tc.double() ** 2).sum().item())
            n_rows += int(t.shape[0] * t.shape[1])
            out.d_s.append(slabs[0])
            out.s_base.append(slabs[1])
            out.t_base.append(slabs[2])
    out.tc_norm2 = tc2
    out.rows_total = n_rows
    diag = prepared["procrustes_diagnostics"]
    for j in range(n_blocks):
        q = out.q0[j]
        out.q_info.append(
            {
                "block": j,
                "source_block": int(pairing.pairing[j]),
                "q_shape": list(q.shape),
                "q_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest(),
                "q_frobenius": float(torch.linalg.norm(q.double()).item()),
                "procrustes_rank": diag[j].get("procrustes_rank"),
                "procrustes_residual_rel": math.sqrt(res2[j] / tc2[j]) if tc2[j] > 0 else None,
            }
        )
    total_bytes = sum(x.numel() * x.element_size() for lst in (out.d_s, out.s_base, out.t_base) for x in lst)
    out.cache_info = {
        "mode": "cpu_resident",
        "dtype": str(cache_dtype).replace("torch.", ""),
        "bytes": int(total_bytes),
        "content": "per calibration batch, stacked over blocks: S_ft-S_base (aligned), S_base (aligned), T_base; "
        "D_j = (S_ft-S_base) Q_j recomputed at every step",
    }
    return out


def act_q_stats(targets: ActTargets, q_list: Sequence[torch.Tensor], device: torch.device) -> list[dict[str, float]]:
    """Per block: ``||Q - Q_procrustes||_F``, relative alignment residual ``||S~ Q - T~||/||T~||`` and ``||D_j||_F``."""
    n_blocks = len(targets.q0)
    res2 = [0.0] * n_blocks
    d2 = [0.0] * n_blocks
    mu_s, mu_t = targets.mu_s.to(device), targets.mu_t.to(device)
    with torch.no_grad():
        for k in range(len(targets.d_s)):
            for j in range(n_blocks):
                q = q_list[j].detach().to(device)
                ds = targets.d_s[k][j].to(device, torch.float32)
                sb = targets.s_base[k][j].to(device, torch.float32)
                tb = targets.t_base[k][j].to(device, torch.float32)
                res2[j] += float((((sb - mu_s[j]) @ q - (tb - mu_t[j])).double() ** 2).sum().item())
                d2[j] += float(((ds @ q).double() ** 2).sum().item())
    return [
        {
            "block": j,
            "q_minus_procrustes_frobenius": float(torch.linalg.norm(q_list[j].detach().cpu() - targets.q0[j]).item()),
            "alignment_residual_rel": math.sqrt(res2[j] / targets.tc_norm2[j]) if targets.tc_norm2[j] > 0 else None,
            "d_frobenius": math.sqrt(d2[j]),
        }
        for j in range(n_blocks)
    ]


def act_displacement_errors(
    model: torch.nn.Module,
    targets: ActTargets,
    q_list: Sequence[torch.Tensor],
    device: torch.device,
) -> list[dict[str, float]]:
    """Per block, on the calibration set: ``||T_j(model) - T_j(base) - D_j|| / ||D_j||`` (ARIADNE's capture, no grad)."""
    from ..rebase.methods._ariadne.capture import iter_capture_tokens

    n_blocks = len(targets.q0)
    err2 = [0.0] * n_blocks
    d2 = [0.0] * n_blocks
    req = {str(j): (j, "boundary") for j in range(n_blocks)}
    for k, got in enumerate(iter_capture_tokens(model, targets.target_batches, req, device, store_device=device)):
        for j in range(n_blocks):
            q = q_list[j].detach().to(device)
            d = targets.d_s[k][j].to(device, torch.float32) @ q
            tb = targets.t_base[k][j].to(device, torch.float32)
            err2[j] += float(((got[str(j)] - tb - d).double() ** 2).sum().item())
            d2[j] += float((d.double() ** 2).sum().item())
    return [
        {"block": j, "rel_error": math.sqrt(err2[j] / d2[j]) if d2[j] > 0 else None, "d_frobenius": math.sqrt(d2[j])}
        for j in range(n_blocks)
    ]


def random_stiefel(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
    """Random point of the rectangular Stiefel manifold (orthonormal columns if rows >= cols, else rows): QR of a Gaussian."""
    r, c = shape
    g = torch.randn((r, c) if r >= c else (c, r), generator=generator, dtype=torch.float64)
    q, rr = torch.linalg.qr(g)
    q = q * torch.sign(torch.diagonal(rr)).clamp_min(-1).where(torch.diagonal(rr) != 0, torch.ones(())).unsqueeze(0)
    return (q if r >= c else q.T).float().contiguous()


class ActObjective:
    """Hooks the residual stream after every block of the student; ``loss_fn(_logits, k)`` is the act loss on batch ``k``.

    ``learned=False``: Q_j fixed to ARIADNE's Procrustes solution; loss = L_disp.
    ``learned=True``: Q_j trainable (orthogonal parametrisation on the rectangular Stiefel manifold; start = random
    point (``q_init='random'``, seeded) or the Procrustes solution) and the loss is
    ``L_disp + align_weight * mean_j ||S~_base Q_j - T~_base||_F^2 / ||T~_base||_F^2``
    (Eq. 1 alignment term; centered banks with ARIADNE's means; the denominator is the whole-bank ``||T~||_F^2``
    scaled to the batch's row share). ``L_disp`` stays the per-element mean square (the same function in both variants).
    """

    def __init__(self, model: torch.nn.Module, targets: ActTargets, device: torch.device, *, learned: bool = False,
                 align_weight: float = 0.0, q_init: str = "random", seed: int = 0) -> None:
        from ..rebase.methods._ariadne.layouts import _layout_for

        self.targets, self.device, self.learned, self.align_weight = targets, device, learned, float(align_weight)
        self.n_blocks = len(targets.q0)
        self._out: dict[int, torch.Tensor] = {}
        layout = _layout_for(None)
        blocks = layout.blocks(model)
        self._handles = []
        for j in range(self.n_blocks):

            def hook(mod, _inp, out, *, _j=j):
                if mod.training and torch.is_grad_enabled():
                    self._out[_j] = out[0] if isinstance(out, tuple) else out

            self._handles.append(layout.block_module(blocks[j]).register_forward_hook(hook))
        self.q_modules: torch.nn.ModuleList | None = None
        if learned:
            from torch.nn.utils.parametrizations import orthogonal

            mods = []
            if q_init not in ("random", "procrustes"):
                raise ValueError("q_init must be 'random' or 'procrustes'")
            gen = torch.Generator().manual_seed(int(seed))
            for q in targets.q0:
                start = random_stiefel(tuple(q.shape), gen) if q_init == "random" else q.clone()
                m = torch.nn.Module()
                m.weight = torch.nn.Parameter(start.to(device))
                orthogonal(m, "weight", orthogonal_map="cayley")
                mods.append(m)
            self.q_modules = torch.nn.ModuleList(mods)
        else:
            self._q_fixed = [q.to(device) for q in targets.q0]

    def q_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.q_modules.parameters()) if self.q_modules is not None else []

    def q_matrices(self) -> list[torch.Tensor]:
        if self.q_modules is not None:
            return [m.weight for m in self.q_modules]
        return self._q_fixed

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._out = {}

    def loss_fn(self, _logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        from ..rebase.methods._shared import _to_tokens

        tg, dev = self.targets, self.device
        kk = k.flatten().tolist() if torch.is_tensor(k) else [k]
        k = int(kk[0])
        sl = slice(int(kk[1]), int(kk[1]) + int(kk[2])) if len(kk) == 3 else slice(None)  # --act-micro-batch rows
        if len(self._out) != self.n_blocks:
            raise RuntimeError("act loss needs a training-mode forward with grad (block outputs were not recorded)")
        q_list = self.q_matrices()
        disp = 0.0
        align = 0.0
        for j in range(self.n_blocks):
            tb = tg.t_base[k][j][sl].to(dev, torch.float32)
            ts = _to_tokens(self._out[j], batch_size=tb.shape[0])
            d = tg.d_s[k][j][sl].to(dev, torch.float32) @ q_list[j]
            disp = disp + ((ts - tb - d) ** 2).mean()
            if self.learned and self.align_weight > 0:
                sc = tg.s_base[k][j][sl].to(dev, torch.float32) - tg.mu_s[j].to(dev)
                tc = tb - tg.mu_t[j].to(dev)
                denom = tg.tc_norm2[j] * (tb.shape[0] * tb.shape[1]) / tg.rows_total
                align = align + ((sc @ q_list[j] - tc) ** 2).sum() / denom
        self._out = {}
        loss = disp / self.n_blocks
        if self.learned and self.align_weight > 0:
            loss = loss + self.align_weight * align / self.n_blocks
        return loss


def act_batches(targets: ActTargets, micro_batch: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``(x, k)`` pairs: the images only (labels dropped by construction) and the batch index into the act cache.

    ``micro_batch > 0`` splits every calibration batch into consecutive micro-batches of that size, keyed
    ``[k, start, size]``; with ``accumulate = batch / micro_batch`` the optimizer step (mean loss over the same 32
    images, one step per calibration batch) is unchanged and only the activation memory is reduced."""
    if micro_batch <= 0:
        return [(b[0], torch.tensor(k)) for k, b in enumerate(targets.target_batches)]
    out = []
    for k, b in enumerate(targets.target_batches):
        n = b[0].shape[0]
        if n % micro_batch:
            raise ValueError(f"--act-micro-batch {micro_batch} must divide the calibration batch size {n}")
        for s0 in range(0, n, micro_batch):
            out.append((b[0][s0:s0 + micro_batch], torch.tensor([k, s0, micro_batch])))
    return out


def _build_source_base(args: argparse.Namespace) -> torch.nn.Module:
    """Frozen source BASE open_clip model (same arch and pretrained weights as the source fine-tuned model), untimed."""
    from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

    cfg = OpenClipBuildConfig(
        model_name=args.teacher_model, pretrained=args.teacher_pretrained, device=args.device, dtype="fp32"
    )
    model = OpenClipClassifier.build(cfg).model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_state_(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            sd[k].copy_(v.to(sd[k].device, sd[k].dtype))


# ---------------------------------------------------------------------------------------------------------------
# data / recipe helpers
# ---------------------------------------------------------------------------------------------------------------


def indices_sha256(indices: Sequence[int]) -> str:
    """Same fingerprint as ``_ariadne.diagnostics`` (sha256 of ``repr`` of the index list)."""
    return hashlib.sha256(repr(list(indices)).encode()).hexdigest()


def split_identity(dataset: Any) -> dict[str, Any]:
    """Deterministic identity of a split: sha256 of its ``sample_ids`` if present, else of the HF split fingerprint."""
    n = len(dataset)
    ids = getattr(dataset, "sample_ids", None)
    if ids is not None:
        return {"kind": "sample_ids", "n": n, "sha256": hashlib.sha256(repr([str(i) for i in ids]).encode()).hexdigest()}
    fingerprint = getattr(getattr(dataset, "split", None), "_fingerprint", None)
    if fingerprint is None:
        return {"kind": "unavailable", "n": n, "sha256": None}
    return {"kind": "hf_fingerprint", "n": n, "fingerprint": str(fingerprint),
            "sha256": hashlib.sha256(repr((str(fingerprint), n)).encode()).hexdigest()}


def seeded_train_loader(loader: Any, seed: int):
    """Same loader with its own shuffling ``torch.Generator(seed)``: val evaluation (global RNG) cannot alter the order."""
    from torch.utils.data import DataLoader

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def calibration_batches(train_loader, *, num_batches: int, seed: int):
    """The Ariadne calibration draw (target view): ``paired_calibration`` replayed with the same loader on both sides.

    Indices depend only on the dataset length, ``seed``, ``num_batches`` and ``batch_size``, so using the target
    loader for both sides reproduces Ariadne's draw exactly (Ariadne additionally materializes the source view).
    """
    from ..rebase.methods._ariadne.capture import paired_calibration

    _, target_batches, meta = paired_calibration(train_loader, train_loader, num_batches=num_batches, seed=seed)
    return target_batches, meta


def calibration_batches_paired(source_train_loader, target_train_loader, *, num_batches: int, seed: int):
    """``paired_calibration`` with a genuinely different source view: ``(source_batches, target_batches, meta)``."""
    from ..rebase.methods._ariadne.capture import paired_calibration

    return paired_calibration(source_train_loader, target_train_loader, num_batches=num_batches, seed=seed)


def native_recipe(target_model: str, task: str) -> dict[str, Any]:
    """Native fine-tuning recipe (yaml of the target architecture) for ``task``."""
    name = {"ViT-L-14": "vision-ft-vitl14-datacomp.yaml", "ViT-B-16": "vision-ft-vitb16-datacomp.yaml"}.get(
        target_model
    )
    if name is None:
        raise ValueError(f"no native recipe yaml for target {target_model!r}")
    cfg = yaml.safe_load((_CONFIG_DIR / name).read_text())
    common = cfg["common"]
    train = common["train"]
    return {
        "yaml": name,
        "epochs": int(cfg["datasets"][task]["train"]["epochs"]),
        "batch_size": int(common["data"]["batch_size"]),
        "accumulate": int(train["accumulate_grad_batches"]),
        "weight_decay": float(train["weight_decay"]),
        "lr_yaml": float(train["lr"]),
        "grad_clip": float(train["grad_clip_norm"]),
        "patience": int(train["early_stopping_patience"]),
        "scheduler": str(train["lr_scheduler"]["name"]),
        "warmup_steps": 500,  # train_vision default (train.lr_scheduler.warmup_steps)
    }


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--task", required=True, help="vision8 task, e.g. EuroSAT")
    p.add_argument("--target-model", default="ViT-L-14")
    p.add_argument("--target-pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--suite", default="vision8")
    p.add_argument("--out-dir", required=True, help="written flat; usable as precomputed_tv_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-5, help="task-arithmetic FT setting; not tuned (repro only)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=89, help="training RNG; default of --split-seed / --calib-seed")
    p.add_argument("--split-seed", type=int, default=None, help="val carve-out seed (default: --seed)")
    p.add_argument("--calib-seed", type=int, default=None, help="calibration draw seed (default: --seed)")
    p.add_argument("--val-size", type=int, default=None, help="unsupported: val is always the pipeline's carve-out")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=10, help="calibration batches (budget / converge_calib)")
    p.add_argument("--batch-size", type=int, default=None, help="default: 32 (all modes)")
    p.add_argument("--accumulate", type=int, default=None, help="default: 4 for converge_full (effective 128, both models), 1 otherwise")
    p.add_argument("--time-budget-s", type=float, default=None, help="budget mode: wall-clock budget (seconds)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="converge_full only: fixed number of optimizer steps (default 2000, task-arithmetic recipe)")
    p.add_argument("--val-every", type=int, default=500, help="converge_full: val diagnostic every N steps")
    p.add_argument("--max-epochs", type=int, default=None, help="default: 200 (converge_calib) / yaml (full)")
    p.add_argument("--patience", type=int, default=None, help="early-stopping patience in epochs (default 10)")
    p.add_argument("--selection", choices=("final", "best_val"), default=None,
                   help="converge modes: final = keep final weights, no early stopping (default for converge_full); "
                   "best_val = early stopping + best-val weights (default for converge_calib)")
    p.add_argument("--scheduler", default=None, help="default: cosine (converge_full), constant otherwise")
    p.add_argument("--warmup-steps", type=int, default=None, help="default: 500 (converge_full), 0 otherwise")
    p.add_argument("--init-delta", default=None, help="{target state_dict name: delta} .pt applied before training")
    p.add_argument("--init-alpha", type=float, default=1.0)
    p.add_argument("--objective", choices=("ce", "kd", "act"), default="ce",
                   help="ce = labelled cross-entropy (default); kd = label-free distillation of the source "
                   "fine-tuned model (teacher) into the target on the same images; act = activation-displacement "
                   "distillation (ARIADNE's per-block D_j/Q_j, fitted end to end, no ridge; needs the --teacher-* flags for the source FT, the "
                   "source base is the same arch with --teacher-pretrained weights)")
    p.add_argument("--teacher-model", default=None, help="kd: open_clip arch of the source model, e.g. ViT-B-16")
    p.add_argument("--teacher-pretrained", default=_TEACHER_PRETRAINED_DEFAULT)
    p.add_argument("--teacher-ckpt", default=None, help="kd: source fine-tuned best.pt (visual.* weights)")
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--act-q", choices=("procrustes", "learned"), default="procrustes",
                   help="act: Q_j fixed to ARIADNE's Procrustes map (default) or learned on the Stiefel manifold")
    p.add_argument("--act-q-init", choices=("procrustes", "random"), default=None,
                   help="act-q learned: start of Q_j; random (default) = QR of a Gaussian on the same Stiefel manifold "
                   "(seeded by --seed), procrustes = ARIADNE's solution")
    p.add_argument("--act-align-weight", type=float, default=0.0,
                   help="act-q learned: weight of the Eq. 1 alignment term (default 0 = displacement loss only)")
    p.add_argument("--act-q-lr", type=float, default=None, help="act-q learned: lr of Q_j (default: --lr); no weight decay")
    p.add_argument("--act-cache-dtype", choices=("fp32", "fp16"), default="fp32", help="act: host cache dtype")
    p.add_argument("--act-cache-max-gb", type=float, default=48.0, help="act: refuse a host cache larger than this")
    p.add_argument("--act-micro-batch", type=int, default=0,
                   help="act: split each calibration batch into micro-batches of this size with gradient accumulation "
                        "(same optimizer step, less activation memory); 0 = off")
    p.add_argument("--num-steps", type=int, default=None, help="steps mode: exact number of optimizer steps (>= 1)")
    p.add_argument("--trainable", choices=TRAINABLE_MODES, default="cproj",
                   help="cproj = MLP c_proj weight+bias of every visual block (default); full = every float visual.* "
                        "parameter (text tower / logit scale / zero-shot head frozen); lora_cproj = LoRA on every visual "
                        "mlp.c_proj.weight; lora_full = LoRA on attn.in_proj_weight, attn.out_proj.weight, mlp.c_fc.weight, "
                        "mlp.c_proj.weight of every visual block (only the LoRA factors are trained; ce/kd only)")
    p.add_argument("--lora-rank", type=int, default=16, help="lora_*: rank r")
    p.add_argument("--lora-alpha", type=float, default=None, help="lora_*: alpha (default = rank, i.e. scale 1)")
    p.add_argument("--micro-batch", type=int, default=0,
                   help="ce/kd: split each calibration batch into micro-batches of this size with gradient accumulation "
                        "(same optimizer step and gradient, less activation memory); 0 = off (act: --act-micro-batch)")
    p.add_argument("--humanize", action="store_true", help="humanized classnames (pipeline default: raw)")
    p.add_argument("--overwrite", action="store_true")
    return p


def _seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_task(args: argparse.Namespace, batch_size: int, split_seed: int):
    """Model, frozen zero-shot head and loaders exactly as the pipeline builds them (target side)."""
    from ..data.templates import get_templates
    from ..data.vision_loaders import build_vision_loaders, load_hf_splits
    from ..eval.datasets.vision8_14_20 import SUITES
    from ..eval.utils import humanize
    from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

    device = torch.device(args.device)
    t_build = time.perf_counter()
    build_cfg = OpenClipBuildConfig(
        model_name=args.target_model, pretrained=args.target_pretrained, device=args.device, dtype="fp32"
    )
    clf = OpenClipClassifier.build(build_cfg)
    hf_path, hf_config, split_map = SUITES[args.suite].resolver(args.task)
    hf_ds = load_hf_splits(hf_path, config=hf_config, requested_splits=tuple(dict.fromkeys(split_map.values())))
    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=clf.preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        val_fraction=args.val_fraction,
        seed=split_seed,
    )
    classnames = [humanize(c) for c in loaders.classnames] if args.humanize else list(loaders.classnames)
    task_cfg = OpenClipBuildConfig(
        model_name=args.target_model,
        pretrained=args.target_pretrained,
        device=args.device,
        dtype="fp32",
        prompt_templates=get_templates(args.task),
    )
    encoder = ImageEncoder(clf).to(device)
    clf.build_zeroshot_text_features(classnames, task_cfg)  # frozen zero-shot head
    return clf, encoder, loaders, time.perf_counter() - t_build


def run(args: argparse.Namespace) -> dict[str, Any]:
    from ..run_logging import _code_fingerprint

    mode = args.mode
    validate_objective_args(args)
    kd = args.objective == "kd"
    act = args.objective == "act"
    if act and mode == "converge_full":
        raise NotImplementedError("--objective act supports budget and converge_calib only")
    if mode == "budget" and not args.time_budget_s:
        raise ValueError("--mode budget requires --time-budget-s")
    if mode == "steps":
        if args.num_steps is None or args.num_steps < 1:
            raise ValueError("--mode steps requires --num-steps >= 1")
        if act:
            raise NotImplementedError("--objective act supports budget and converge_calib only")
    elif args.num_steps is not None:
        raise ValueError("--num-steps only applies to --mode steps")
    if args.micro_batch < 0:
        raise ValueError("--micro-batch must be >= 0")
    if args.micro_batch and (act or mode == "converge_full"):
        raise ValueError("--micro-batch applies to ce/kd on calibration batches (act: --act-micro-batch; not converge_full)")
    if args.val_size is not None:
        raise ValueError(
            "--val-size is not supported: early-stopping val must equal the pipeline's alpha-selection val "
            "(the val_fraction carve-out of the test split)"
        )
    split_seed = args.seed if args.split_seed is None else args.split_seed
    calib_seed = args.seed if args.calib_seed is None else args.calib_seed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_name = canonical_method_name("direct_residual")
    tv_path = out_dir / f"{args.task}_{method_name}_transported_native.pt"
    json_path = out_dir / f"{args.task}_cproj_calib.json"
    if (tv_path.exists() or json_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{tv_path} / {json_path} exist (use --overwrite)")

    recipe = native_recipe(args.target_model, args.task)
    full = mode == "converge_full"
    # converge_full: 32 x 4 = effective 128 for BOTH models (the B/16 yaml says 128 x 1; 16 GB GPUs need micro-batch 32)
    batch_size = args.batch_size if args.batch_size is not None else 32
    accumulate = args.accumulate if args.accumulate is not None else (4 if full else 1)
    if mode in ("budget", "steps") and accumulate != 1:
        raise ValueError(f"--mode {mode} takes one optimizer step per calibration batch (--accumulate must be 1)")
    if accumulate < 1:
        raise ValueError("--accumulate must be >= 1")
    scheduler = args.scheduler or (recipe["scheduler"] if full else "constant")
    warmup = args.warmup_steps if args.warmup_steps is not None else (recipe["warmup_steps"] if full else 0)
    selection = args.selection if args.selection is not None else ("final" if full else "best_val")
    if mode in ("budget", "steps"):
        selection = None
    patience = args.patience if args.patience is not None else (recipe["patience"] if full else 10)
    if full:
        if args.max_epochs is not None:
            raise ValueError("--max-epochs does not apply to converge_full (fixed --max-steps recipe)")
        if selection != "final":
            raise ValueError("converge_full uses a fixed number of steps and keeps the final weights (--selection final)")
        max_steps = args.max_steps if args.max_steps is not None else 2000
        max_epochs = None
    else:
        if args.max_steps is not None:
            raise ValueError("--max-steps only applies to --mode converge_full")
        max_steps = None
        max_epochs = args.max_epochs if args.max_epochs is not None else 200
        if mode == "steps":
            if args.max_epochs is not None:
                raise ValueError("--max-epochs does not apply to --mode steps (use --num-steps)")
            max_epochs = None
    device = torch.device(args.device)
    _seed_all(args.seed)

    clf, encoder, loaders, build_seconds = _build_task(args, batch_size, split_seed)

    if full:  # own generator: training order independent of the global RNG (val passes draw from it)
        import dataclasses

        seeded = seeded_train_loader(loaders.train, args.seed)
        if dataclasses.is_dataclass(loaders):  # VisionLoaders is frozen
            loaders = dataclasses.replace(loaders, train=seeded)
        else:
            loaders.train = seeded
    if "val_size" in loaders.meta:
        raise ValueError("loaders were built with val_size: val would be carved from train (unsupported)")
    model = clf.model
    lora_mode = args.trainable in LORA_MODES
    keys = visual_float_keys(model) if args.trainable in ("full", "lora_full") else cproj_keys(model)
    val_zero_shot = float(encoder.top1(loaders.val, str(device)))
    base = snapshot(model, keys)

    init_keys: list[str] = []
    init_sha = None
    if args.init_delta:
        init = torch.load(args.init_delta, map_location="cpu", weights_only=True)
        init_keys = list(init)
        base.update({k: v for k, v in snapshot(model, [k for k in init_keys if k not in base]).items()})
        apply_init_delta(model, init, args.init_alpha)
        init_sha = hashlib.sha256(Path(args.init_delta).read_bytes()).hexdigest()
    extra_keys = [k for k in init_keys if k not in keys]
    val_start = float(encoder.top1(loaders.val, str(device))) if init_keys else val_zero_shot

    # ---- kd: teacher build/load, own zero-shot head, its val top-1 (diagnostic): all outside the timed region
    teacher = teacher_loaders = None
    teacher_view = teacher_val = None
    source_base = None
    if kd or act:
        teacher, teacher_loaders, teacher_view = _build_teacher(args, clf, loaders, batch_size, split_seed)
        if act:
            source_base = _build_source_base(args)
        if full and teacher_loaders is not None:
            raise ValueError("kd in converge_full needs identical teacher/target preprocessing (no separate view)")
        teacher_val = float(teacher.top1((teacher_loaders or loaders).val, str(device)))
        print(f"teacher view={teacher_view} val top-1={teacher_val:.4f}")
    loss_fn = (lambda s_logits, t_logits: kd_loss(s_logits, t_logits, args.kd_temperature)) if kd else None
    targets = objective = None
    act_seconds = None

    # ---- timed region: calibration-data materialization + trainer setup + training steps (model/text head are
    # already on device; val evaluation, best-state snapshots and saving are excluded in every mode).
    recorder = PhaseCostRecorder(device)
    result: dict[str, Any]
    with recording(recorder):
        mark = recorder.mark()
        _sync(device)
        t0 = time.perf_counter()
        calib_meta = None
        if mode in ("budget", "converge_calib", "steps"):
            if act:
                # Q_j, source/target activation collection and the dS/S_base/T_base cache: all inside the timed region,
                # computed on the UNTOUCHED target (an --init-delta is swapped out for the capture, restored exactly).
                t_act = time.perf_counter()
                held = snapshot(model, list(base)) if init_keys else None
                if held is not None:
                    _load_state_(model, base)
                try:
                    targets = build_act_targets(
                        model,
                        source_base,
                        teacher.clip_model.model,
                        (teacher_loaders or loaders).train,
                        loaders.train,
                        num_batches=args.num_batches,
                        seed=calib_seed,
                        device=device,
                        cache_dtype=torch.float32 if args.act_cache_dtype == "fp32" else torch.float16,
                        max_cache_gb=args.act_cache_max_gb,
                    )
                finally:
                    if held is not None:
                        _load_state_(model, held)
                objective = ActObjective(
                    model, targets, device, learned=args.act_q == "learned", align_weight=args.act_align_weight,
                    q_init=args.act_q_init or "random", seed=args.seed,
                )
                loss_fn = objective.loss_fn
                q_initial = [q.detach().clone() for q in objective.q_matrices()]  # the real starting Q (random or Procrustes)
                batches, calib_meta = act_batches(targets, args.act_micro_batch), targets.calibration
                if args.act_micro_batch > 0:  # same optimizer step per calibration batch, half the activation memory
                    accumulate = targets.target_batches[0][0].shape[0] // args.act_micro_batch
                # the source models are only needed for the capture (dS is cached): free their GPU memory for training
                # (L/14 target at batch 32 otherwise runs out of memory on 16 GB)
                teacher.cpu()
                source_base.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _sync(device)
                act_seconds = time.perf_counter() - t_act
                steps_per_epoch = math.ceil(len(batches) / accumulate)
            elif teacher_loaders is not None:  # separate teacher view of the same indices
                t_views, batches, calib_meta = calibration_batches_paired(
                    teacher_loaders.train, loaders.train, num_batches=args.num_batches, seed=calib_seed
                )
            else:
                batches, calib_meta = calibration_batches(loaders.train, num_batches=args.num_batches, seed=calib_seed)
                t_views = batches
            if kd:  # teacher logits: computed ONCE (inside the timed region), labels dropped from the batches
                batches = kd_batches(batches, compute_teacher_logits(teacher, t_views, device))
                del t_views
            if not act:
                if args.micro_batch > 0:  # same optimizer step per calibration batch, lower activation memory
                    k_micro = batches[0][0].shape[0] // args.micro_batch
                    batches = micro_split(batches, args.micro_batch)
                    accumulate_train = accumulate * k_micro
                else:
                    accumulate_train = accumulate
                steps_per_epoch = math.ceil(len(batches) / accumulate_train)
            else:
                accumulate_train = accumulate
        else:
            batches = None
            steps_per_epoch = 0
            accumulate_train = accumulate
        lora = (
            attach_lora(model, args.trainable, rank=args.lora_rank, alpha=args.lora_alpha, seed=args.seed)
            if lora_mode
            else None
        )
        opt, sched, tinfo = configure_cproj_training(
            encoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup,
            scheduler_name=scheduler,
            total_steps=(
                max_steps if full else args.num_steps if mode == "steps" else max_epochs * steps_per_epoch
            ),
            device=device,
            trainable=args.trainable,
            lora=lora,
        )
        print(f"trainable: {tinfo['trainable_params']} params in {tinfo['trainable_tensors']} tensors")
        if act and args.act_q == "learned":
            from .schedulers import build_lr_scheduler

            q_lr = args.lr if args.act_q_lr is None else args.act_q_lr
            opt.add_param_group({"params": objective.q_parameters(), "lr": q_lr, "weight_decay": 0.0})
            sched = build_lr_scheduler(
                opt, name=scheduler, base_lrs=[args.lr, q_lr], warmup_length=warmup,
                steps=max(1, int(max_epochs * steps_per_epoch)),
            )
        if mode == "steps":
            result = train_fixed_steps(
                encoder,
                batches,
                opt,
                sched,
                num_steps=int(args.num_steps),
                t0=t0,
                grad_clip=args.grad_clip,
                device=device,
                loss_fn=loss_fn,
                accumulate=accumulate_train,
            )
            _sync(device)
            timed_total = time.perf_counter() - t0
            cuda_peak, host_peak = recorder.peaks_since(mark)
        elif mode == "budget":
            result = train_budget(
                encoder,
                batches,
                opt,
                sched,
                time_budget_s=float(args.time_budget_s),
                t0=t0,
                grad_clip=args.grad_clip,
                device=device,
                loss_fn=loss_fn,
                accumulate=accumulate_train,
            )
            _sync(device)
            timed_total = time.perf_counter() - t0
            cuda_peak, host_peak = recorder.peaks_since(mark)
        else:
            _sync(device)
            setup_seconds = time.perf_counter() - t0
            setup_cuda, setup_host = recorder.peaks_since(mark)
            if full:
                result = train_steps(
                    encoder,
                    loaders.train,
                    loaders.val,
                    opt,
                    sched,
                    max_steps=max_steps,
                    eval_every=args.val_every,
                    accumulate=accumulate,
                    grad_clip=args.grad_clip,
                    device=device,
                    start_val=val_start,
                    recorder=recorder,
                    teacher=teacher,
                    kd_temperature=args.kd_temperature,
                )
            else:
                result = train_converge(
                    encoder,
                    (lambda: batches) if batches is not None else (lambda: loaders.train),
                    loaders.val,
                    opt,
                    sched,
                    keys=[] if lora_mode else keys,  # LoRA: best-epoch snapshot is the factors (extra_module)
                    max_epochs=max_epochs,
                    patience=patience,
                    accumulate=accumulate_train,
                    grad_clip=args.grad_clip,
                    device=device,
                    start_val=val_start,
                    selection=selection,
                    recorder=recorder,
                    loss_fn=loss_fn,
                    extra_module=lora.module_list if lora_mode else (objective.q_modules if act else None),
                )
            timed_total = setup_seconds + result["train_seconds"]
            cuda_peak = max(setup_cuda, result["train_cuda_peak_bytes"])
            host_peak = max(setup_host, result["train_host_peak_bytes"])

    act_summary = None
    if act:
        objective.close()
        q_final = [q.detach() for q in objective.q_matrices()]
        q_start = act_q_stats(targets, q_initial, device)
        q_end = act_q_stats(targets, q_final, device)
        errors = act_displacement_errors(model, targets, q_final, device)
        learned_q_path = None
        if args.act_q == "learned":
            learned_q_path = out_dir / f"{args.task}_act_Q_learned.pt"
            torch.save([q.cpu() for q in q_final], learned_q_path)
        act_summary = {
            "q_mode": args.act_q,
            "learned_q_file": str(learned_q_path) if learned_q_path else None,
            "source": {
                "model": args.teacher_model,
                "ft_ckpt": args.teacher_ckpt,
                "ft_ckpt_sha256": file_sha256(args.teacher_ckpt),
                "base_pretrained": args.teacher_pretrained,
                "view": teacher_view,
                "val_top1_ft_head": teacher_val,
            },
            "depth_map": list(targets.pairing),
            "source_depth": targets.source_depth,
            "target_depth": len(targets.pairing),
            "cache": targets.cache_info,
            "setup_seconds_act_capture": act_seconds,
            "centering": "D_j uncentered; free bias = trainable c_proj.bias (ARIADNE ridge intercept)",
            "loss": "L_disp = mean_j mean_{rows,dims} ||T_j(student)-T_j(base)-D_j||^2"
            + (
                "; + align_weight * mean_j ||S~Q_j-T~||^2/||T~||^2" if args.act_q == "learned" and args.act_align_weight > 0 else ""
            ),
            "align_weight": args.act_align_weight if args.act_q == "learned" else None,
            "q_init": (args.act_q_init or "random") if args.act_q == "learned" else "procrustes (fixed)",
            "q_lr": (args.lr if args.act_q_lr is None else args.act_q_lr) if args.act_q == "learned" else None,
            "q_check_procrustes": targets.q_info,
            "q_stats_start": q_start,
            "q_stats_end": q_end,
            "final_displacement_error_per_block": errors,
            "final_displacement_error_mean": sum(e["rel_error"] for e in errors if e["rel_error"] is not None)
            / max(1, len(errors)),
            "note": "final per-block error = cheap log of this training run on the calibration set (final weights)",
        }

    val_alpha1 = float(encoder.top1(loaders.val, str(device)))
    if lora_mode:  # explicit merged update scale*B@A (no W+BA-W rounding); base weights are restored untouched below
        delta = lora_delta_dict(lora, base, keys)
        tinfo = dict(tinfo)
        tinfo["delta_frobenius_norm"] = math.sqrt(sum(float((v.double() ** 2).sum()) for v in delta.values()))
        tinfo["delta_nonzero_tensors"] = sum(int(torch.count_nonzero(v) > 0) for v in delta.values())
        lora.detach()
    else:
        delta = extract_delta(model, base, keys + extra_keys)
    for k, v in delta.items():
        if not torch.isfinite(v).all():
            raise FloatingPointError(f"non-finite delta in {k}")
    torch.save(delta, tv_path)

    fp = _code_fingerprint()
    summary = {
        "task": args.task,
        "mode": mode,
        "lr": args.lr,
        "config": {
            **{k: v for k, v in vars(args).items()},
            "batch_size": batch_size,
            "accumulate": accumulate,
            "scheduler": scheduler,
            "warmup_steps": warmup,
            "patience": None if full else patience,
            "selection": selection,
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "effective_batch": batch_size * accumulate,
            "native_recipe_yaml_reference_only": recipe,
            "recipe_used": {
                "max_steps": max_steps,
                "batch_size": batch_size,
                "accumulate": accumulate,
                "effective_batch": batch_size * accumulate,
                "warmup_steps": warmup,
                "scheduler": scheduler,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "selection": selection,
            },
            "lr_protocol": (f"fixed {args.lr:g} (task-arithmetic FT setting), not tuned" if args.trainable not in LORA_MODES else f"LoRA lr {args.lr:g} (grid point; per-task selection on val_acc_alpha1 by the queue)"),
        },
        "git_commit": fp.get("git_commit"),
        "git_dirty": fp.get("git_dirty"),
        "trainable": tinfo,
        "calibration_indices_sha256": indices_sha256(calib_meta["indices"]) if calib_meta else None,
        "calibration_meta": {k: v for k, v in (calib_meta or {}).items() if k != "indices"},
        "steps": result["steps"],
        "epochs": result["epochs"],
        "timed_region": {
            "definition": _TIMED_DEFINITION[mode]
            + (
                "; act objective: Q_j computation, source/target activation collection and the dS/S_base/T_base "
                "caching are inside the region"
                if act
                else ""
            ),
            "wall_seconds": timed_total,
            "train_seconds": result["train_seconds"],
            "cuda_peak_allocated_bytes": cuda_peak,
            "host_peak_rss_bytes": host_peak,
            "host_peak_method": recorder.host_peak_method,
        },
        "build_seconds_untimed": build_seconds,
        "eval_seconds_untimed": result.get("eval_seconds"),
        "split": {
            "meta": loaders.meta,
            "sizes": dict(loaders.sizes),
            "val_fraction": args.val_fraction,
            "split_seed": split_seed,
            "calib_seed": calib_seed,
            "train_identity": split_identity(loaders.train.dataset),
            "val_identity": split_identity(loaders.val.dataset),
            "test_identity": split_identity(loaders.test.dataset),
        },
        "train_loss_steps": result["train_loss_steps"],
        "epoch_log": result.get("epoch_log"),
        "selection": selection,
        "max_steps": max_steps,
        "effective_batch": batch_size * accumulate,
        "passes_over_data": result["epochs"],
        "val_log": result.get("val_log"),
        "val_acc_final": result.get("val_acc_final"),
        "best_epoch": result.get("best_epoch"),
        "best_val_acc": result.get("best_val_acc"),
        "val_acc_zero_shot": val_zero_shot,
        "val_acc_start": val_start,
        "val_acc_alpha1": val_alpha1,
        "init_delta": {"path": args.init_delta, "sha256": init_sha, "alpha": args.init_alpha} if init_keys else None,
        "delta_file": str(tv_path),
        "delta_keys": len(delta),
        "target": {"model": args.target_model, "pretrained": args.target_pretrained},
        "objective": args.objective,
        "distillation": {
            "teacher_model": args.teacher_model,
            "teacher_pretrained": args.teacher_pretrained,
            "teacher_ckpt": args.teacher_ckpt,
            "teacher_ckpt_sha256": file_sha256(args.teacher_ckpt),
            "temperature": args.kd_temperature,
            "loss": "T^2 * KL(softmax(teacher/T) || softmax(student/T)), batch mean, no labels",
            "teacher_view": teacher_view,
            "teacher_logits": "cached once on the calibration batches, inside the timed region"
            if not full
            else "computed on the fly per batch (inside the timed region)",
            "teacher_val_top1": teacher_val,
        }
        if kd
        else None,
    }
    if act_summary is not None:
        summary["act"] = act_summary
    json_path.write_text(json.dumps(summary, indent=1))
    print(f"[{args.task}/{mode}] steps={result['steps']} val(alpha=1)={val_alpha1:.4f} -> {tv_path}")
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
