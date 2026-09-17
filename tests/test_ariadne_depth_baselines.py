"""Regression tests for the two ARIADNE depth baselines.

Both baselines answer "what does the extra depth actually buy?" from a
different side:

* ``inserted_block_mode='residual_identity'`` inserts a block whose output
  projections are zero, so the expanded model computes exactly the original
  function and the inserted depth carries no computation at all.
* ``transport_activation_mode='interpolate_neighbors'`` keeps ARIADNE's
  structural initialization but hands Theseus/BiCo the midpoint of the two
  original blocks' activations instead of the inserted block's own forward
  pass.

The assertions here are exact rather than approximate: a residual-identity
block is an algebraic identity, and an interpolated activation is an exact
midpoint, so either property failing is a defect and not numerical drift.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.block_extension import (
    BlockExtensionConfig,
    build_extension_layout,
    resolve_block_extension_config,
    run_block_extension,
)
from merge_and_rebase.rebase.methods.theseus import (
    InterpolatedBlockActivations,
    collect_activations,
)


class _TinyAttn(nn.Module):
    """Minimal stand-in for CLIP's MultiheadAttention with a fused in_proj."""

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * 0.1)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key=None, value=None, **kwargs) -> torch.Tensor:
        del key, value, kwargs
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = q.shape[-1] ** -0.5
        weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        return self.out_proj(weights @ v)


class _TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(torch.relu(self.c_fc(x)))


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _TinyAttn(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _TinyMLP(dim)

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs):
        del attn_mask, kwargs
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _TinyVisual(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_TinyBlock(width) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 4, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        x = self.ln_post(x)
        return x.mean(dim=1)


class _TinyModel(nn.Module):
    def __init__(self, in_dim: int = 6, width: int = 8, depth: int = 3):
        super().__init__()
        self.visual = _TinyVisual(in_dim=in_dim, width=width, depth=depth)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _make_loader(n_samples: int = 16, in_dim: int = 6, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n_samples, in_dim)
    y = torch.zeros(n_samples, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

_BASELINE_PARAMS = {
    "residual_identity": {"inserted_block_mode": "residual_identity"},
    "residual_identity_inert": {"inserted_block_mode": "residual_identity_inert"},
    "interpolated_activations": {"transport_activation_mode": "interpolate_neighbors"},
}


def _extend(
    *,
    depth: int = 3,
    blocks_to_add: int = 2,
    strategy: str = "interpolate_per_weight",
    inserted_block_mode: str = "ariadne",
    seed: int = 0,
) -> tuple[torch.nn.Module, torch.nn.Module, dict, torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(seed)
    source_base = _TinyModel(depth=depth)
    source_ft = _TinyModel(depth=depth)
    # Snapshot both endpoints before the extender edits them in place: the
    # function-preservation assertions compare against these.
    pristine_base = _TinyModel(depth=depth)
    pristine_base.load_state_dict(source_base.state_dict())
    pristine_ft = _TinyModel(depth=depth)
    pristine_ft.load_state_dict(source_ft.state_dict())

    cfg = BlockExtensionConfig(
        blocks_to_add=blocks_to_add,
        extension_strategy=strategy,
        n_batches_act=1,
        skip_correction=True,
        skip_final_ln=True,
        inserted_block_mode=inserted_block_mode,
        verbose=False,
        show_progress=False,
    )
    layout: dict = {}
    run_block_extension(
        source_base_model=source_base,
        source_ft_model=source_ft,
        calibration_loader=_make_loader(),
        target_layers_total=None,
        config=cfg,
        device="cpu",
        layout_out=layout,
    )
    return source_base, source_ft, layout, pristine_base, pristine_ft


@pytest.mark.parametrize("params", list(_BASELINE_PARAMS.values()), ids=list(_BASELINE_PARAMS))
def test_depth_baselines_require_skip_correction(params: dict) -> None:
    """A fitted correction and a baseline inserted block are mutually exclusive."""
    with pytest.raises(ValueError, match="skip_correction"):
        resolve_block_extension_config(
            {"block_extension_enabled": True, "block_extension_params": dict(params)}
        )

    _, cfg = resolve_block_extension_config(
        {
            "block_extension_enabled": True,
            "block_extension_params": {**params, "skip_correction": True},
        }
    )
    for key, value in params.items():
        assert getattr(cfg, key) == value


def test_block_extension_config_defaults_to_ariadne() -> None:
    _, cfg = resolve_block_extension_config({})
    assert cfg.inserted_block_mode == "ariadne"
    assert cfg.transport_activation_mode == "model"


@pytest.mark.parametrize("field", ["inserted_block_mode", "transport_activation_mode"])
def test_block_extension_config_rejects_unknown_baseline(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        resolve_block_extension_config(
            {
                "block_extension_enabled": True,
                "block_extension_params": {field: "nonsense", "skip_correction": True},
            }
        )


def test_residual_identity_preserves_the_original_function_exactly() -> None:
    """A zero-projection block is an identity, so depth changes but outputs do not."""
    source_base, _, layout, pristine_base, _ = _extend(inserted_block_mode="residual_identity")

    inputs = torch.randn(5, 6)
    with torch.no_grad():
        extended = source_base.encode_image(inputs)
        original = pristine_base.encode_image(inputs)

    assert len(source_base.visual.transformer.resblocks) == 5
    assert len(pristine_base.visual.transformer.resblocks) == 3
    assert torch.equal(extended, original)
    assert len(layout["inserted_blocks"]) == 2


def test_residual_identity_zeroes_both_output_projections() -> None:
    source_base, source_ft, layout, _, _ = _extend(inserted_block_mode="residual_identity")

    inserted_positions = {block["position"] for block in layout["inserted_blocks"]}
    for model in (source_base, source_ft):
        for position, block in enumerate(model.visual.transformer.resblocks):
            projections = (block.attn.out_proj, block.mlp.c_proj)
            if position in inserted_positions:
                for projection in projections:
                    assert torch.count_nonzero(projection.weight) == 0
                    assert torch.count_nonzero(projection.bias) == 0
            else:
                assert torch.count_nonzero(projections[0].weight) > 0


def test_residual_identity_task_vector_is_inert_at_inserted_projections() -> None:
    """The inserted block stays an identity after any transported delta is added.

    Both endpoints carry zero output projections, so the task vector is exactly
    zero there and no merge weighting can reintroduce computation at that depth.
    """
    source_base, source_ft, layout, _, _ = _extend(inserted_block_mode="residual_identity")
    base_sd = source_base.state_dict()
    ft_sd = source_ft.state_dict()

    for block in layout["inserted_blocks"]:
        position = block["position"]
        for suffix in (
            f"visual.transformer.resblocks.{position}.attn.out_proj.weight",
            f"visual.transformer.resblocks.{position}.attn.out_proj.bias",
            f"visual.transformer.resblocks.{position}.mlp.c_proj.weight",
            f"visual.transformer.resblocks.{position}.mlp.c_proj.bias",
        ):
            assert torch.count_nonzero(ft_sd[suffix] - base_sd[suffix]) == 0


def test_extension_layout_records_the_bracketing_original_blocks() -> None:
    """bottom-top/spread inserts a descendant of block 0, then one of block 1."""
    _, _, layout, _, _ = _extend(blocks_to_add=2)

    assert layout["final_depth"] == 5
    assert layout["inserted_blocks"] == (
        {
            "position": 1,
            "source_orig_idx": 0,
            "neighbour_orig_idx": 1,
            "source_position": 0,
            "neighbour_position": 2,
        },
        {
            "position": 3,
            "source_orig_idx": 1,
            "neighbour_orig_idx": 2,
            "source_position": 2,
            "neighbour_position": 4,
        },
    )
    assert layout["original_positions"] == {0: 0, 1: 2, 2: 4}


def test_extension_layout_clamps_the_last_block_to_itself() -> None:
    """The top block has no successor, matching the weight midpoint's clamp."""
    chain = [
        {"orig_idx": 0, "inserted": False},
        {"orig_idx": 1, "inserted": False},
        {"orig_idx": 1, "inserted": True, "neighbour_orig_idx": 1},
    ]
    layout = build_extension_layout(chain)

    assert layout["inserted_blocks"] == (
        {
            "position": 2,
            "source_orig_idx": 1,
            "neighbour_orig_idx": 1,
            "source_position": 1,
            "neighbour_position": 1,
        },
    )


def test_interpolated_plan_replaces_only_inserted_positions() -> None:
    plan = InterpolatedBlockActivations.from_extension_layout(
        {"inserted_blocks": ({"position": 1, "source_position": 0, "neighbour_position": 2},)}
    )
    store = {
        "transformer.resblocks.0.attn.out_proj": torch.ones(2, 4),
        "transformer.resblocks.1.attn.out_proj": torch.full((2, 4), 9.0),
        "transformer.resblocks.2.attn.out_proj": torch.full((2, 4), 3.0),
        # Guards against prefix matching treating block 10 as block 1.
        "transformer.resblocks.10.attn.out_proj": torch.full((2, 4), 7.0),
    }
    plan.apply(store)

    assert torch.equal(store["transformer.resblocks.1.attn.out_proj"], torch.full((2, 4), 2.0))
    assert torch.equal(store["transformer.resblocks.0.attn.out_proj"], torch.ones(2, 4))
    assert torch.equal(store["transformer.resblocks.2.attn.out_proj"], torch.full((2, 4), 3.0))
    assert torch.equal(store["transformer.resblocks.10.attn.out_proj"], torch.full((2, 4), 7.0))


def test_interpolated_plan_is_per_component() -> None:
    """Each component reads its own neighbour banks, not one shared block bank."""
    plan = InterpolatedBlockActivations.from_extension_layout(
        {"inserted_blocks": ({"position": 1, "source_position": 0, "neighbour_position": 2},)}
    )
    store = {}
    for position, value in ((0, 1.0), (1, 99.0), (2, 5.0)):
        for component in ("ln_1", "attn.q_proj", "attn.out_proj", "ln_2", "mlp.c_fc", "mlp.c_proj"):
            offset = len(component)
            store[f"transformer.resblocks.{position}.{component}"] = torch.full((2, 3), value + offset)
    plan.apply(store)

    for component in ("ln_1", "attn.q_proj", "attn.out_proj", "ln_2", "mlp.c_fc", "mlp.c_proj"):
        offset = len(component)
        assert torch.equal(
            store[f"transformer.resblocks.1.{component}"],
            torch.full((2, 3), 0.5 * ((1.0 + offset) + (5.0 + offset))),
        )


def test_interpolated_plan_requires_both_neighbours() -> None:
    plan = InterpolatedBlockActivations.from_extension_layout(
        {"inserted_blocks": ({"position": 1, "source_position": 0, "neighbour_position": 2},)}
    )
    store = {
        "transformer.resblocks.0.attn.out_proj": torch.ones(2, 4),
        "transformer.resblocks.1.attn.out_proj": torch.ones(2, 4),
    }
    with pytest.raises(KeyError, match="transformer.resblocks.2.attn.out_proj"):
        plan.apply(store)


def test_interpolated_plan_needs_at_least_one_inserted_block() -> None:
    with pytest.raises(ValueError, match="at least one inserted block"):
        InterpolatedBlockActivations.from_extension_layout({"inserted_blocks": ()})


def test_interpolated_plan_fingerprint_separates_layouts() -> None:
    """The fingerprint reaches the activation cache key, so it must discriminate."""
    first = InterpolatedBlockActivations.from_extension_layout(
        {"inserted_blocks": ({"position": 1, "source_position": 0, "neighbour_position": 2},)}
    )
    second = InterpolatedBlockActivations.from_extension_layout(
        {"inserted_blocks": ({"position": 3, "source_position": 2, "neighbour_position": 4},)}
    )
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() == InterpolatedBlockActivations(entries=first.entries).fingerprint()


def _raw_source_rows(registry, key: str) -> torch.Tensor:
    source, _ = registry[key].rows()
    assert source is not None
    return source


def test_collect_activations_substitutes_the_inserted_block_bank() -> None:
    """End to end: Theseus sees the neighbour midpoint at the inserted position."""
    source_base, _, layout, _, _ = _extend(blocks_to_add=2)
    torch.manual_seed(1)
    target_model = _TinyModel(depth=5)

    images = torch.randn(8, 6)
    labels = torch.zeros(8, dtype=torch.long)
    loader_kwargs = dict(batch_size=4, shuffle=False)

    def _loader() -> DataLoader:
        return DataLoader(TensorDataset(images, labels), **loader_kwargs)

    plan = InterpolatedBlockActivations.from_extension_layout(layout)
    common = dict(device="cpu", seq_align="interpolate2d", n_batches=2, seed=0, store_raw=True)

    reference = collect_activations(source_base, target_model, _loader(), _loader(), **common)
    baseline = collect_activations(
        source_base, target_model, _loader(), _loader(), source_activation_plan=plan, **common
    )

    assert set(reference) == set(baseline)
    inserted = {block["position"] for block in layout["inserted_blocks"]}
    checked = 0
    for key in reference:
        if not key.startswith("transformer.resblocks."):
            continue
        position = int(key.split(".")[2])
        suffix = key.split(".", 3)[3]
        if position not in inserted:
            assert torch.equal(_raw_source_rows(reference, key), _raw_source_rows(baseline, key))
            continue
        block = next(b for b in layout["inserted_blocks"] if b["position"] == position)
        left = f"transformer.resblocks.{block['source_position']}.{suffix}"
        right = f"transformer.resblocks.{block['neighbour_position']}.{suffix}"
        expected = 0.5 * (_raw_source_rows(reference, left) + _raw_source_rows(reference, right))
        assert torch.allclose(_raw_source_rows(baseline, key), expected, atol=0, rtol=0)
        assert not torch.equal(_raw_source_rows(baseline, key), _raw_source_rows(reference, key))
        checked += 1

    assert checked > 0


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_inert_insertion_zeroes_the_whole_inserted_task_vector() -> None:
    """The inert arm removes the inserted block from the task vector entirely.

    ``residual_identity`` still carries the source block's input-side deltas
    (ln, q/k/v, c_fc) at the inserted position. The inert arm gives both
    endpoints the same block, so nothing at that depth is transported, and the
    merged target keeps its own weights there.
    """
    source_base, source_ft, layout, _, _ = _extend(inserted_block_mode="residual_identity_inert")
    base_sd = source_base.state_dict()
    ft_sd = source_ft.state_dict()

    for block in layout["inserted_blocks"]:
        prefix = f"visual.transformer.resblocks.{block['position']}."
        inserted_keys = [key for key in base_sd if key.startswith(prefix)]
        assert inserted_keys
        for key in inserted_keys:
            assert torch.count_nonzero(ft_sd[key] - base_sd[key]) == 0, key


def test_inert_insertion_preserves_both_endpoint_functions_exactly() -> None:
    source_base, source_ft, _, pristine_base, pristine_ft = _extend(
        inserted_block_mode="residual_identity_inert"
    )

    inputs = torch.randn(5, 6)
    with torch.no_grad():
        assert torch.equal(source_base.encode_image(inputs), pristine_base.encode_image(inputs))
        # The FT endpoint's inserted blocks come from the base endpoint, but a
        # zero-projection block is an identity in any model, so the FT function
        # is preserved too.
        assert torch.equal(source_ft.encode_image(inputs), pristine_ft.encode_image(inputs))


def test_inert_insertion_shares_the_base_endpoint_with_residual_identity() -> None:
    """The two arms must fit byte-identical transport maps.

    Theseus and BiCo both calibrate on the source *base* endpoint, so an
    identical base endpoint means identical activations and identical fitted
    maps. That is what licenses reading the accuracy difference between the two
    arms as the effect of the transported delta alone.
    """
    identity_base, identity_ft, _, _, _ = _extend(inserted_block_mode="residual_identity")
    inert_base, inert_ft, _, _, _ = _extend(inserted_block_mode="residual_identity_inert")

    assert _state_hash(identity_base) == _state_hash(inert_base)
    # The FT endpoints must differ, or the ablation changed nothing at all.
    assert _state_hash(identity_ft) != _state_hash(inert_ft)


def test_inert_insertion_is_extension_only() -> None:
    with pytest.raises(ValueError, match="extension baseline"):
        _extend(depth=5, blocks_to_add=-2, inserted_block_mode="residual_identity_inert")
