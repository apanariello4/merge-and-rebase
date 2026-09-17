from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from merge_and_rebase.eval.block_extension import BlockExtensionConfig
from merge_and_rebase.eval.block_extension_llm import (
    DecoderBlockExtender,
    run_block_extension_llm,
)


class DummyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DummyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = attn @ v
        return self.o_proj(out)


class DummyMLP(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DummyDecoderBlock(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.input_layernorm = DummyRMSNorm(dim)
        self.self_attn = DummyAttention(dim)
        self.post_attention_layernorm = DummyRMSNorm(dim)
        self.mlp = DummyMLP(dim, intermediate)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class DummyDecoderBody(nn.Module):
    def __init__(self, n_layers: int, dim: int, intermediate: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderBlock(dim, intermediate) for _ in range(n_layers)])
        self.norm = DummyRMSNorm(dim)


class DummyDecoderModel(nn.Module):
    def __init__(self, n_layers: int, dim: int = 16, intermediate: int = 32, vocab: int = 100):
        super().__init__()
        self.config = type("Config", (), {"model_type": "dummy", "hidden_size": dim, "intermediate_size": intermediate, "num_hidden_layers": n_layers, "num_attention_heads": 4, "num_key_value_heads": None})()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.model = DummyDecoderBody(n_layers, dim, intermediate)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embed_tokens(input_ids)
        # Mirrors HF decoders, which iterate `self.layers[: config.num_hidden_layers]`.
        for layer in self.model.layers[: self.config.num_hidden_layers]:
            x = layer(x)
        x = self.model.norm(x)
        return type("Output", (), {"logits": self.lm_head(x)})()


class DummyFamilyAdapter:
    name = "dummy"

    def metadata(self, model):
        cfg = model.config
        from merge_and_rebase.rebase.model_families.base import ModelFamilyMetadata
        return ModelFamilyMetadata(
            family=self.name,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
        )

    def transport_scope(self, model):
        return model.model

    def transportable_keys(self, state_dict):
        return {k for k in state_dict if "layers." in k or k == "model.norm.weight"}

    def param_to_module(self, model):
        scope = self.transport_scope(model)
        out = {}
        for name, module in scope.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                key = f"{name}.{pname}" if name else pname
                out[key] = name
                out[f"model.{key}"] = name
        return out

    def iter_blocks(self, model):
        yield from self.transport_scope(model).layers

    def block_count(self, model):
        return len(self.transport_scope(model).layers)

    def extract_calibration_batch(self, batch):
        if isinstance(batch, dict):
            return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
        return {}

    def excluded_keys(self):
        return {"embed_tokens", "lm_head"}


def _make_calibration_loader(vocab: int = 100, batch_size: int = 2, seq_len: int = 8, n_batches: int = 1):
    from torch.utils.data import DataLoader, Dataset

    class PromptDataset(Dataset):
        def __init__(self):
            self.data = torch.randint(0, vocab, (batch_size * n_batches, seq_len))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ids = self.data[idx]
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return DataLoader(PromptDataset(), batch_size=batch_size, collate_fn=_collate)


class TestDuplicationSchedule:
    def test_bottom_top_spread(self):
        sched = DecoderBlockExtender._build_duplication_schedule(4, 2, "bottom-top", "spread")
        assert len(sched) == 2
        assert all(0 <= s < 4 for s in sched)

    def test_top_bottom_spread(self):
        sched = DecoderBlockExtender._build_duplication_schedule(4, 2, "top-bottom", "spread")
        assert len(sched) == 2
        assert all(0 <= s < 4 for s in sched)

    def test_clump(self):
        sched = DecoderBlockExtender._build_duplication_schedule(4, 3, "bottom-top", "clump")
        assert sched == [0, 0, 0]

    def test_zero_needed(self):
        assert DecoderBlockExtender._build_duplication_schedule(4, 0, "bottom-top", "spread") == []

    def test_spread_covers_the_whole_model(self):
        # Qwen2.5-1.5B -> 3B: 8 blocks added to a 28-layer model. This used to
        # return [0..7], doubling the bottom third and leaving 8-27 untouched.
        sched = DecoderBlockExtender._build_duplication_schedule(28, 8, "bottom-top", "spread")
        assert len(sched) == 8
        assert len(set(sched)) == 8
        assert sched == sorted(sched)
        assert sched[0] == 0
        gaps = [b - a for a, b in zip(sched, sched[1:], strict=False)]
        assert max(gaps) - min(gaps) <= 1
        assert sched[-1] >= 28 - max(gaps) - 1  # reaches the top of the model
        # never the final block: duplicating it right before the output norm is
        # far more destructive than any other single placement
        assert 27 not in sched

    def test_spread_top_bottom_runs_from_the_top(self):
        sched = DecoderBlockExtender._build_duplication_schedule(28, 8, "top-bottom", "spread")
        assert len(set(sched)) == 8
        assert sched == sorted(sched, reverse=True)
        assert sched[0] == 26  # highest anchor, but never the final block
        assert 27 not in sched

    def test_spread_covers_every_block_when_doubling(self):
        # n_needed >= curr_layers: every block gets a duplicate, final one included
        assert DecoderBlockExtender._build_duplication_schedule(6, 12, "bottom-top", "spread") == [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
        ]

    def test_spread_unchanged_at_one_duplicate_per_block(self):
        assert DecoderBlockExtender._build_duplication_schedule(6, 6, "bottom-top", "spread") == list(range(6))

    def test_spread_mod_still_gives_the_old_prefix_schedule(self):
        # kept as the escape hatch for reproducing pre-fix runs
        assert DecoderBlockExtender._build_duplication_schedule(28, 8, "bottom-top", "spread_mod") == list(range(8))


class TestCollapseSchedule:
    def test_bottom_top_spread(self):
        sched = DecoderBlockExtender._build_collapse_schedule(6, 2, "bottom-top", "spread")
        assert len(sched) == 2
        assert all(0 <= s < 5 for s in sched)

    def test_spread_covers_the_whole_model(self):
        sched = DecoderBlockExtender._build_collapse_schedule(28, 8, "bottom-top", "spread")
        assert len(sched) == 8
        assert len(set(sched)) == 8
        assert sched[0] == 0
        assert all(0 <= a <= 26 for a in sched)  # max anchor is curr_layers - 2
        gaps = [b - a for a, b in zip(sched, sched[1:], strict=False)]
        assert max(gaps) - min(gaps) <= 1

    def test_clump(self):
        sched = DecoderBlockExtender._build_collapse_schedule(6, 2, "bottom-top", "clump")
        assert sched == [0, 0]

    def test_zero_to_remove(self):
        assert DecoderBlockExtender._build_collapse_schedule(6, 0, "bottom-top", "spread") == []


class TestDecoderBlockExtender:
    @pytest.fixture
    def small_model(self):
        return DummyDecoderModel(n_layers=3, dim=16, intermediate=32)

    @pytest.fixture
    def family_adapter(self):
        return DummyFamilyAdapter()

    @pytest.fixture
    def calib_loader(self):
        return _make_calibration_loader(vocab=100, batch_size=2, seq_len=8, n_batches=1)

    def test_extend_interpolate_per_weight(self, small_model, family_adapter, calib_loader):
        model_base = small_model
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy="interpolate_per_weight",
            blocks_to_add=2,
            insertion_order="bottom-top",
            extension_density="spread",
            skip_correction=True,
        )
        assert final_depth == 5
        assert family_adapter.block_count(model_base) == 5
        assert family_adapter.block_count(model_ft) == 5

    @pytest.mark.parametrize("strategy", ["interpolate", "duplicate"])
    def test_non_per_weight_strategies_are_rejected(
        self, small_model, family_adapter, calib_loader, strategy
    ):
        """The bare strategies were deprecated; only the per-weight cascade is supported."""
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            small_model, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        with pytest.raises(ValueError, match="non per-weight"):
            extender.extend_and_calibrate(
                loader=calib_loader,
                n_batches=1,
                strategy=strategy,
                blocks_to_add=2,
                skip_correction=True,
            )

    def test_extend_per_weight_skip_correction(self, small_model, family_adapter, calib_loader):
        model_base = small_model
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy="per_weight",
            blocks_to_add=1,
            insertion_order="bottom-top",
            extension_density="spread",
            skip_correction=True,
        )
        assert final_depth == 4

    @pytest.mark.parametrize(
        ("strategy", "expected_mode"),
        [
            ("interpolate_per_weight", "cascade"),
            ("duplicate_per_weight", "duplicate"),
        ],
    )
    def test_vision_per_weight_strategy_aliases_extend(
        self, small_model, family_adapter, calib_loader, strategy, expected_mode
    ):
        model_base = small_model
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy=strategy,
            target_layers_total=4,
            skip_correction=True,
        )
        assert final_depth == 4
        assert expected_mode in {"cascade", "duplicate"}

    def test_extend_per_weight_with_correction(self, small_model, family_adapter, calib_loader):
        model_base = small_model
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy="per_weight",
            blocks_to_add=1,
            insertion_order="bottom-top",
            extension_density="spread",
            skip_correction=False,
            n_cascade_iters=1,
        )
        assert final_depth == 4

    def test_shrink(self, small_model, family_adapter, calib_loader):
        model_base = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
        model_ft = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy="shrink",
            target_layers_total=3,
            insertion_order="bottom-top",
            extension_density="spread",
            skip_correction=True,
        )
        assert final_depth == 3

    @pytest.mark.parametrize("strategy", ["interpolate_per_weight", "duplicate_per_weight"])
    def test_vision_per_weight_strategy_aliases_shrink(self, family_adapter, calib_loader, strategy):
        model_base = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
        model_ft = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy=strategy,
            target_layers_total=4,
            skip_correction=True,
        )
        assert final_depth == 4

    def test_no_extension_needed(self, small_model, family_adapter, calib_loader):
        model_base = small_model
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        extender = DecoderBlockExtender(
            model_base, model_ft, family_adapter, device="cpu", verbose=False, show_progress=False
        )
        final_depth = extender.extend_and_calibrate(
            loader=calib_loader,
            n_batches=1,
            strategy="interpolate_per_weight",
            target_layers_total=3,
            skip_correction=True,
        )
        assert final_depth == 3


@pytest.mark.parametrize("strategy", ["interpolate_per_weight", "duplicate_per_weight"])
def test_shared_reverse_fits_ft_and_applies_to_base_for_extension(monkeypatch, strategy):
    base = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
    ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
    adapter = DummyFamilyAdapter()
    extender = DecoderBlockExtender(base, ft, adapter, device="cpu", verbose=False, show_progress=False)
    correction_calls: list[str] = []
    apply_calls: list[nn.Module] = []

    monkeypatch.setattr(extender, "capture_reference_inputs", lambda loader, n_batches: None)
    monkeypatch.setattr(extender, "_capture_component_references", lambda loader, n_batches: None)

    def fake_correct(model_name, model, *args, **kwargs):
        correction_calls.append(model_name)
        kwargs["lmc_store"]["sentinel"] = (torch.eye(16), torch.zeros(16))

    monkeypatch.setattr(extender, "_correct_block_weights_cascade", fake_correct)
    monkeypatch.setattr(extender, "_apply_block_corrections", lambda model, *_: apply_calls.append(model))
    extender.extend_and_calibrate(
        loader=object(), n_batches=1, strategy=strategy, target_layers_total=4,
        skip_correction=False, lmc_mode="shared_reverse",
    )
    assert correction_calls == ["ft"]
    assert apply_calls == [base]


@pytest.mark.parametrize("strategy", ["interpolate_per_weight", "duplicate_per_weight"])
def test_shared_reverse_fits_ft_and_applies_to_base_for_shrink(monkeypatch, strategy):
    base = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
    ft = DummyDecoderModel(n_layers=5, dim=16, intermediate=32)
    adapter = DummyFamilyAdapter()
    extender = DecoderBlockExtender(base, ft, adapter, device="cpu", verbose=False, show_progress=False)
    correction_calls: list[str] = []
    apply_calls: list[nn.Module] = []

    monkeypatch.setattr(extender, "capture_reference_inputs", lambda loader, n_batches: None)
    monkeypatch.setattr(extender, "_capture_component_references", lambda loader, n_batches: None)

    def fake_correct(model_name, model, *args, **kwargs):
        correction_calls.append(model_name)
        kwargs["lmc_store"]["sentinel"] = (torch.eye(16), torch.zeros(16))

    monkeypatch.setattr(extender, "_correct_collapsed_block_weights_cascade", fake_correct)
    monkeypatch.setattr(extender, "_apply_block_corrections", lambda model, *_: apply_calls.append(model))
    extender.extend_and_calibrate(
        loader=object(), n_batches=1, strategy=strategy, target_layers_total=4,
        skip_correction=False, lmc_mode="shared_reverse",
    )
    assert correction_calls == ["ft"]
    assert apply_calls == [base]


class TestRunBlockExtensionLLM:
    def test_run_entrypoint(self):
        model_base = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        adapter = DummyFamilyAdapter()
        loader = _make_calibration_loader(vocab=100, batch_size=2, seq_len=8, n_batches=1)
        config = BlockExtensionConfig(
            blocks_to_add=2,
            extension_strategy="interpolate_per_weight",
            skip_correction=True,
            n_batches_act=1,
            verbose=False,
            show_progress=False,
        )
        final_depth = run_block_extension_llm(
            source_base_model=model_base,
            source_ft_model=model_ft,
            calibration_loader=loader,
            target_layers_total=5,
            config=config,
            family_adapter=adapter,
            device="cpu",
        )
        assert final_depth == 5
        assert len(model_base.model.layers) == 5
        # A longer ModuleList alone is inert: HF decoders slice by
        # config.num_hidden_layers, so the appended blocks never run while
        # state_dict still reports them.
        assert model_base.config.num_hidden_layers == 5
        assert model_ft.config.num_hidden_layers == 5

    def test_extended_model_executes_every_layer(self):
        model_base = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        model_ft = DummyDecoderModel(n_layers=3, dim=16, intermediate=32)
        adapter = DummyFamilyAdapter()
        loader = _make_calibration_loader(vocab=100, batch_size=2, seq_len=8, n_batches=1)
        config = BlockExtensionConfig(
            blocks_to_add=2,
            extension_strategy="interpolate_per_weight",
            skip_correction=True,
            n_batches_act=1,
            verbose=False,
            show_progress=False,
        )
        run_block_extension_llm(
            source_base_model=model_base,
            source_ft_model=model_ft,
            calibration_loader=loader,
            target_layers_total=5,
            config=config,
            family_adapter=adapter,
            device="cpu",
        )

        executed: list[int] = []
        handles = [
            layer.register_forward_hook(lambda _m, _i, _o, idx=idx: executed.append(idx))
            for idx, layer in enumerate(model_base.model.layers)
        ]
        try:
            model_base(input_ids=torch.randint(0, 100, (1, 8)))
        finally:
            for handle in handles:
                handle.remove()

        assert executed == [0, 1, 2, 3, 4]


def _record_ridge_targets(monkeypatch) -> list[torch.Tensor]:
    """Capture every least-squares target the cascade fits, in call order.

    The fitted map is replaced by the identity so the corrections are no-ops and
    only the target construction is under test.
    """
    recorded: list[torch.Tensor] = []

    def fake_fit_ridge(A, T, lambda_reg=1e-6, ridge_id=0.0, ridge_target=None):
        recorded.append(T.detach().clone())
        dim = int(T.shape[1])
        return torch.eye(dim), torch.zeros(dim)

    monkeypatch.setattr(DecoderBlockExtender, "_fit_ridge", staticmethod(fake_fit_ridge))
    return recorded


def _record_captures(monkeypatch) -> dict[str, list[torch.Tensor]]:
    """Record what each activation capture returned, without changing it."""
    captured: dict[str, list[torch.Tensor]] = {}
    real_component = DecoderBlockExtender._capture_component_output
    real_input = DecoderBlockExtender._capture_single_input

    def wrapped_component(self, model, block_idx, component, loader, n_batches):
        out = real_component(self, model, block_idx, component, loader, n_batches)
        captured.setdefault(component, []).append(out.detach().clone())
        return out

    def wrapped_input(self, model, target, loader, n_batches):
        out = real_input(self, model, target, loader, n_batches)
        captured.setdefault("__block_input__", []).append(out.detach().clone())
        return out

    monkeypatch.setattr(DecoderBlockExtender, "_capture_component_output", wrapped_component)
    monkeypatch.setattr(DecoderBlockExtender, "_capture_single_input", wrapped_input)
    return captured


# The per-block cascade fits nine components in a fixed order.
_CASCADE_STEPS = 9
_O_PROJ_STEP = 4
_DOWN_PROJ_STEP = 8


def test_inserted_block_targets_its_source_blocks_own_activations(monkeypatch) -> None:
    """An inserted block reproduces its source block's component activations.

    The residual-aware form belongs to the collapse path. Using it here
    subtracts the block input a second time -- step 5 has already absorbed it --
    which drives the inserted block towards inverting its own source block
    instead of repeating it.
    """
    recorded = _record_ridge_targets(monkeypatch)
    extender = DecoderBlockExtender(
        DummyDecoderModel(n_layers=3, dim=16, intermediate=32),
        DummyDecoderModel(n_layers=3, dim=16, intermediate=32),
        DummyFamilyAdapter(),
        device="cpu",
        verbose=False,
        show_progress=False,
    )
    extender.extend_and_calibrate(
        loader=_make_calibration_loader(vocab=100, batch_size=2, seq_len=8, n_batches=1),
        n_batches=1,
        strategy="interpolate_per_weight",
        blocks_to_add=1,
        insertion_order="bottom-top",
        extension_density="spread",
        skip_correction=False,
    )

    assert len(recorded) >= _CASCADE_STEPS
    last_pass = recorded[-_CASCADE_STEPS:]
    refs = extender.reference_inputs["ft"]

    for step, suffix in ((_O_PROJ_STEP, ".attn_output"), (_DOWN_PROJ_STEP, ".down_proj_output")):
        target = last_pass[step]
        candidates = [v for k, v in refs.items() if k.endswith(suffix)]
        assert candidates, f"no reference activations captured for {suffix}"
        assert any(
            torch.allclose(target, ref[: target.shape[0]], atol=1e-6) for ref in candidates
        ), f"target for {suffix} is not a source block's activation"


def test_collapsed_block_target_subtracts_the_full_pre_mlp_stream(monkeypatch) -> None:
    """The collapse path keeps its residual-aware target, but the pre-MLP stream
    is the block input *plus* the already-corrected attention output. Omitting
    the attention term leaves the fit short by exactly that contribution.
    """
    recorded = _record_ridge_targets(monkeypatch)
    captured = _record_captures(monkeypatch)
    extender = DecoderBlockExtender(
        DummyDecoderModel(n_layers=4, dim=16, intermediate=32),
        DummyDecoderModel(n_layers=4, dim=16, intermediate=32),
        DummyFamilyAdapter(),
        device="cpu",
        verbose=False,
        show_progress=False,
    )
    extender.extend_and_calibrate(
        loader=_make_calibration_loader(vocab=100, batch_size=2, seq_len=8, n_batches=1),
        n_batches=1,
        strategy="interpolate_per_weight",
        target_layers_total=3,
        insertion_order="bottom-top",
        extension_density="spread",
        skip_correction=False,
    )

    # The attention contribution must be captured at all; before this was fixed
    # the collapse cascade never asked for it.
    assert "attn" in captured, "collapse cascade never captured the attention output"

    target = recorded[-1]
    cur_input = captured["__block_input__"][-1]
    cur_attn = captured["attn"][-1]
    n = int(target.shape[0])
    refs = extender.reference_inputs["ft"]
    span_outputs = [v for k, v in refs.items() if k.endswith(".input") and not k.startswith("final")]

    assert any(
        torch.allclose(target, ref[:n] - cur_input[:n] - cur_attn[:n], atol=1e-5)
        for ref in span_outputs
    ), "collapse target does not subtract both the block input and the attention output"
