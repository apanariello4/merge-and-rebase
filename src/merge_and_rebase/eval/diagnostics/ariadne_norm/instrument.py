"""Instrumented sufficient statistics: the official fit runs unchanged while N3/N5/N7 inputs are collected."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

import torch

from merge_and_rebase.rebase.methods._ariadne.fit import ResidualSufficientStatistics

from .analysis import row_concentration, spectral_budget, split_half_stability

Tensor = torch.Tensor


class FitRecorder:
    """Collects one task's per-position analyses from `InstrumentedStatistics.solve`.

    ``token_index_fn(k, n_rows)`` returns the within-sequence token index of each of the ``n_rows`` rows that batch
    ``k`` contributes (0 = CLS for vision, first content token for decoders).
    """

    def __init__(
        self,
        *,
        ridge_estimator: str,
        ridge_relative: float,
        exact_form: bool,
        row_positions: Collection[int],
        token_index_fn: Callable[[int, int], Tensor],
    ) -> None:
        self.ridge_estimator = ridge_estimator
        self.ridge_relative = ridge_relative
        self.exact_form = exact_form
        self.row_positions = set(row_positions)
        self.token_index_fn = token_index_fn
        self.records: dict[int, dict[str, Any]] = {}
        self.arrays: dict[str, Tensor] = {}

    def factory(self, position: int, component: str, device) -> InstrumentedStatistics:
        return InstrumentedStatistics(device=device, position=position, component=component, recorder=self)


class InstrumentedStatistics(ResidualSufficientStatistics):
    """``update``/``solve`` are the parent's, unchanged; side channels feed odd/even copies and optional row banks."""

    def __init__(self, device, *, position: int, component: str, recorder: FitRecorder) -> None:
        super().__init__(device=device)
        self.position = position
        self.component = component
        self.recorder = recorder
        self.halves = [ResidualSufficientStatistics(device=device), ResidualSufficientStatistics(device=device)]
        self.n_updates = 0
        self.keep_rows = position in recorder.row_positions
        self.h_rows: list[Tensor] = []
        self.e_rows: list[Tensor] = []
        self.token_rows: list[Tensor] = []

    def update(self, h, e, t_in, t_out, row_mask=None) -> None:
        super().update(h, e, t_in, t_out, row_mask=row_mask)
        if row_mask is not None:
            raise ValueError("instrumented statistics expect pre-masked rows (row_mask=None), as Pass B passes them")
        k = self.n_updates
        self.halves[k % 2].update(h, e, t_in, t_out)
        if self.keep_rows:
            self.h_rows.append(h.detach().float().cpu())
            self.e_rows.append(e.detach().float().cpu())
            self.token_rows.append(self.recorder.token_index_fn(k, int(h.shape[0])))
        self.n_updates += 1

    def solve(self, **kwargs):
        x, diag = super().solve(**kwargs)
        rec = self.recorder
        if kwargs.get("exact_form", True) is not True:
            raise ValueError("the norm analysis assumes exact_form=True")
        if kwargs.get("ridge_estimator") != rec.ridge_estimator:
            raise ValueError("recorder and fit disagree on ridge_estimator")
        record: dict[str, Any] = {"position": self.position, "component": self.component}
        spectral, arrays = spectral_budget(self, ridge_estimator=rec.ridge_estimator, ridge_relative=rec.ridge_relative)
        # Internal consistency: the analysis reproduces the solver's own norms.
        spectral["solver_correction_norm"] = float(diag["correction_norm"])
        spectral["solver_bias_norm"] = float(diag["bias_norm"])
        spectral["solver_residual_norm_after"] = float(diag["residual_norm_after"])
        spectral["solver_lam"] = float(diag["ridge"])
        record["n3"] = spectral
        record["n7_regression"] = split_half_stability(
            self, self.halves, ridge_estimator=rec.ridge_estimator, ridge_relative=rec.ridge_relative
        )
        if self.keep_rows:
            record["n5"] = row_concentration(
                torch.cat(self.h_rows),
                torch.cat(self.e_rows),
                torch.cat(self.token_rows),
                self._t_out.cpu(),
                ridge_estimator=rec.ridge_estimator,
                ridge_relative=rec.ridge_relative,
                exact_form=True,
                device=self.device,
            )
            self.h_rows, self.e_rows, self.token_rows = [], [], []
        for name, value in arrays.items():
            rec.arrays[f"pos{self.position}.{self.component}.{name}"] = value
        rec.records[self.position] = record
        self.halves = []
        return x, diag
