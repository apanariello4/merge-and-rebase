"""Text calibration corpus for the LLM rebase run (resolved lazily, cached, recorded in the summary)."""

from __future__ import annotations

from typing import Any

from ...data.llm_calibration import resolve_calibration_texts, tokenization_stats


class TextCalibrationCache:
    """Resolved on first use: building it from an lm-harness task has to index the task registry, which is far
    too expensive to pay for on a run that never collects activations at all."""

    def __init__(
        self,
        *,
        prompts: Any,
        calibration_dataset_cfg: Any,
        block_extension_cfg: Any,
        harness_tasks: Any,
        n_sequences_cfg: Any,
        n_calib_batches: int,
        calib_batch_size: int,
        calib_max_length: int,
        seed: int,
        include_target: bool,
        tokenizer: Any,
    ) -> None:
        self._prompts = prompts
        self._calibration_dataset_cfg = calibration_dataset_cfg
        self._block_extension_cfg = block_extension_cfg
        self._harness_tasks = harness_tasks
        self._n_sequences_cfg = n_sequences_cfg
        self._n_calib_batches = n_calib_batches
        self._calib_batch_size = calib_batch_size
        self._calib_max_length = calib_max_length
        self._seed = seed
        self._include_target = include_target
        self._tokenizer = tokenizer
        self._cache: list[Any] = []

    def get(self) -> Any:
        if not self._cache:
            block_extension_cfg = self._block_extension_cfg
            resolved = resolve_calibration_texts(
                prompts=self._prompts,
                calibration_dataset=(
                    self._calibration_dataset_cfg
                    or block_extension_cfg.calibration_dataset
                    or block_extension_cfg.calibration_task
                ),
                calibration_split=str(block_extension_cfg.calibration_split),
                harness_tasks=list(self._harness_tasks),
                n_sequences=(
                    int(self._n_sequences_cfg)
                    if self._n_sequences_cfg is not None
                    else max(1, self._n_calib_batches) * self._calib_batch_size
                ),
                seed=self._seed,
                include_target=self._include_target,
            )
            for note in resolved.notes:
                print(f"Calibration note: {note}")
            print(f"Calibration corpus: {resolved.describe()}")
            self._cache.append(resolved)
        return self._cache[0]

    def provenance(self) -> dict[str, Any] | None:
        """Additive summary record (None when no calibration corpus was ever resolved)."""
        if not self._cache:
            return None
        record = self._cache[0].provenance()
        record["calibration_batch_size"] = self._calib_batch_size
        record["calibration_max_length"] = self._calib_max_length
        try:
            record["tokenization"] = tokenization_stats(self._tokenizer, self._cache[0].texts, self._calib_max_length)
        except Exception as exc:  # noqa: BLE001 - provenance must never fail a finished run
            record["tokenization"] = {"error": f"{type(exc).__name__}: {exc}"}
        return record
