"""Architecture layouts (CLIP ViT vs. HF decoder) and shape primitives for Ariadne."""

from __future__ import annotations

import torch
from torch import nn

from ....models.vision_utils import _encode_image
from ...model_families import accessors as _fam
from .._shared import _interp_2d_tokens

# --- architecture abstraction -------------------------------------------------
# Only this layer touches the model's module tree: the two layouts name where blocks live, which
# projection writes the residual, how to run a batch, and the state-dict key. family_adapter=None
# selects the original CLIP paths verbatim (vision behaviour unchanged).


class _VisionLayout:
    """CLIP ViT: the original, unchanged code paths."""

    name = "vision"

    def blocks(self, model):
        return model.visual.transformer.resblocks

    def block_count(self, model):
        return len(model.visual.transformer.resblocks)

    def proj_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.mlp.c_proj

    def attn_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.attn

    def attn_proj_module(self, block):
        inner = block.block if hasattr(block, "block") else block
        return inner.attn.out_proj

    def block_module(self, block):
        return block.block if hasattr(block, "block") else block

    def forward(self, model, batch, device):
        _encode_image(model, batch[0].to(device))

    def batch_size(self, batch):
        return len(batch[0])

    def proj_key(self, pos, *, prefixed):
        key = f"transformer.resblocks.{pos}.mlp.c_proj.weight"
        return f"visual.{key}" if prefixed else key

    def attn_proj_key(self, pos, *, prefixed):
        key = f"transformer.resblocks.{pos}.attn.out_proj.weight"
        return f"visual.{key}" if prefixed else key

    def component_key(self, pos, component, *, prefixed):
        if component == "mlp.c_proj":
            return self.proj_key(pos, prefixed=prefixed)
        if component == "attn.out_proj":
            return self.attn_proj_key(pos, prefixed=prefixed)
        if component == "mlp.c_fc":
            key = f"transformer.resblocks.{pos}.mlp.c_fc.weight"
            return f"visual.{key}" if prefixed else key
        if component in _PACKED_QKV_SLICE:
            # q/k/v share one packed parameter; the caller writes into (and
            # reads out of) a single row slice of it.
            key = f"transformer.resblocks.{pos}.attn.in_proj_weight"
            return f"visual.{key}" if prefixed else key
        raise ValueError(f"Unsupported completion component {component!r}")

    def component_scale_module(self, block, component):
        """The LayerScale on this component's write into the residual stream.

        ``ls_1`` scales the attention path and ``ls_2`` the MLP path; either may
        be ``nn.Identity``. Both must enter the fit's output map, not be applied
        afterwards.
        """
        inner = self.block_module(block)
        name = "ls_2" if component == "mlp.c_proj" else "ls_1"
        return getattr(inner, name, nn.Identity())

    def mlp_in_module(self, block):
        """``mlp.c_fc``, the MLP's first (pre-GELU) linear projection."""
        inner = block.block if hasattr(block, "block") else block
        return inner.mlp.c_fc


class _DecoderLayout:
    """HF decoder: mlp.down_proj is the residual-writing projection, the
    analogue of CLIP's mlp.c_proj (both are the block's final output map)."""

    name = "decoder"

    def __init__(self, family_adapter):
        self.family_adapter = family_adapter

    def blocks(self, model):
        return _fam.layers(self.family_adapter, model)

    def block_count(self, model):
        return self.family_adapter.block_count(model)

    def proj_module(self, block):
        return _fam.residual_writer(self.family_adapter, block)

    def attn_module(self, block):
        return _fam.attn_module(self.family_adapter, block)

    def attn_proj_module(self, block):
        """``self_attn.o_proj`` is the decoder analogue of CLIP's attn.out_proj.

        Implemented for symmetry with the vision path and **untested on this
        iteration**: no decoder campaign has run the two-component completion.
        """
        return _fam.attn_output(self.family_adapter, block)

    def block_module(self, block):
        return block

    def forward(self, model, batch, device):
        # Backbone-only, no labels / LM head, use_cache=False (hooks sit on the layers inside the backbone).
        self.family_adapter.calibration_forward(model, batch, device)

    def batch_size(self, batch):
        inputs = self.family_adapter.extract_calibration_batch(batch)
        return int(inputs["input_ids"].shape[0])

    def proj_key(self, pos, *, prefixed):
        # The decoder state dict is already "model.layers.N..."; there is no
        # second prefix the way vision has "visual.".
        return _fam.param_key(
            self.family_adapter, pos, _fam.canonical_components(self.family_adapter)["mlp.c_proj"] + ".weight"
        )

    def attn_proj_key(self, pos, *, prefixed):
        return _fam.param_key(
            self.family_adapter, pos, _fam.canonical_components(self.family_adapter)["attn.out_proj"] + ".weight"
        )

    def component_key(self, pos, component, *, prefixed):
        if component == "mlp.c_proj":
            return self.proj_key(pos, prefixed=prefixed)
        if component == "attn.out_proj":
            return self.attn_proj_key(pos, prefixed=prefixed)
        raise NotImplementedError(f"{component!r} is only supported on the vision (stock nn.MultiheadAttention) layout")

    def component_scale_module(self, block, component):
        """Decoder blocks carry no LayerScale; the write is unscaled."""
        return nn.Identity()

    def mlp_in_module(self, block):
        raise NotImplementedError("mlp_input capture is vision-only")


def _layout_for(family_adapter):
    return _VisionLayout() if family_adapter is None else _DecoderLayout(family_adapter)


#: Capture kinds understood by ``capture_tokens``; ``*_input`` kinds hook the module's input, the rest
#: its output. ``attn_input`` is ln_1's output (shared by q/k/v); ``mlp_input`` is the input of
#: ``mlp.c_fc``; ``block_input`` is the pristine block input ``X_j^0`` (same module as ``"boundary"``,
#: input rather than output). All vision-layout only.
_CAPTURE_KINDS = frozenset(
    {
        "boundary",
        "c_proj",
        "c_proj_input",
        "attn_proj",
        "attn_proj_input",
        "attn_input",
        "mlp_input",
        "block_input",
    }
)


_INPUT_CAPTURE_KINDS = frozenset({"c_proj_input", "attn_proj_input", "attn_input", "mlp_input", "block_input"})


_ATTN_CAPTURE_KINDS = frozenset({"attn_proj", "attn_proj_input"})


_ATTN_QUERY_KINDS = frozenset({"attn_input"})


#: Which capture kind supplies each component's regression features.
COMPONENT_INPUT_KIND = {
    "attn.out_proj": "attn_proj_input",
    "mlp.c_proj": "c_proj_input",
    "attn.q_proj": "attn_input",
    "attn.k_proj": "attn_input",
    "attn.v_proj": "attn_input",
    "mlp.c_fc": "mlp_input",
}


#: Row index of each packed-QKV component within nn.MultiheadAttention's
#: in_proj_weight [3d,d] / in_proj_bias [3d] (F._in_projection_packed's
#: convention: w_q, w_k, w_v stacked along dim 0).
_PACKED_QKV_SLICE = {"attn.q_proj": 0, "attn.k_proj": 1, "attn.v_proj": 2}


def _component_weight_bias(shim, block, component):
    """Return ``(weight, bias_or_None, row_slice_or_None)`` for one component's
    own linear map, i.e. the raw (pre-LayerScale) projection it applies to its
    captured input: ``A = X @ weight.T + bias``.

    ``row_slice`` is non-``None`` only for the packed q/k/v components, whose
    weight/bias are a row range of the attention's ``in_proj_weight`` /
    ``in_proj_bias`` rather than a standalone parameter.
    """
    inner = shim.block_module(block)
    if component == "mlp.c_proj":
        module = inner.mlp.c_proj
        return module.weight, module.bias, None
    if component == "mlp.c_fc":
        module = inner.mlp.c_fc
        return module.weight, module.bias, None
    if component == "attn.out_proj":
        module = inner.attn.out_proj
        return module.weight, module.bias, None
    if component in _PACKED_QKV_SLICE:
        attn = inner.attn
        if not isinstance(attn, nn.MultiheadAttention):
            raise NotImplementedError(f"{component} requires a stock nn.MultiheadAttention module")
        d = int(attn.embed_dim)
        s = _PACKED_QKV_SLICE[component]
        row_slice = slice(s * d, (s + 1) * d)
        weight = attn.in_proj_weight[row_slice]
        bias = attn.in_proj_bias[row_slice] if attn.in_proj_bias is not None else None
        return weight, bias, row_slice
    raise ValueError(f"Unsupported completion component {component!r}")


def _rows(batches):
    return torch.cat([b.reshape(-1, b.shape[-1]) for b in batches], dim=0)


def _aligned(source, target):
    if len(source) != len(target):
        raise ValueError("Reference batch counts do not match")
    result = []
    for a, b in zip(source, target, strict=True):
        if a.shape[0] != b.shape[0]:
            raise ValueError("Reference image counts do not match")
        result.append(_interp_2d_tokens(a, b.shape[1]))
    return result


def _component_effective_out(shim, target_model, pos, component, width) -> torch.Tensor:
    """Build the ``effective_out`` (identity, or LayerScale-diagonal) a component's
    solve is pushed through, exactly as ``_fit_all_positions_independent`` inlined it.
    """
    identity_out = torch.eye(width, dtype=torch.float32)
    scale_module = shim.component_scale_module(shim.blocks(target_model)[pos], component)
    if isinstance(scale_module, nn.Identity):
        return identity_out
    scale = getattr(scale_module, "gamma", None)
    if scale is None or scale.ndim != 1 or scale.shape[0] != width:
        raise ValueError("Unsupported non-diagonal target LayerScale")
    return identity_out * scale.detach().cpu().float().unsqueeze(0)


def _family_bias_key(key: str) -> str | None:
    """Bias key paired with ``key``, or ``None`` if that projection has no bias."""
    if key.endswith("in_proj_weight"):
        return key[: -len("in_proj_weight")] + "in_proj_bias"
    if key.endswith(".weight"):
        return f"{key[: -len('.weight')]}.bias"
    return None
