"""Pins the chunk-1 (bitwise-neutral) streaming-capture refactor of
``capture_tokens``/``iter_capture_tokens`` in ``target_informed_runtime``.

``capture_tokens`` is now a thin wrapper accumulating ``iter_capture_tokens``'s
per-batch yields; both share ``_register_capture_hooks`` for hook resolution.
This module is self-contained -- no cross-test imports -- matching the rest of
this suite.
"""

from collections import OrderedDict

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.eval.target_informed_runtime import capture_tokens, iter_capture_tokens


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_1 = torch.nn.Identity()
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.ls_1(self.attn(x)) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _loader(n=8, batch_size=2):
    return DataLoader(TensorDataset(torch.randn(n, 5, 4), torch.arange(n)), batch_size=batch_size, shuffle=False)


def _model(depth=3, width=6):
    torch.manual_seed(3)
    return _Model(width, depth).eval()


def _requests():
    return {
        "b0_boundary": (0, "boundary"),
        "b1_block_input": (1, "block_input"),
        "b1_mlp_input": (1, "mlp_input"),
        "b2_attn_proj_input": (2, "attn_proj_input"),
        "b2_attn_proj": (2, "attn_proj"),
    }


def test_capture_tokens_equals_concatenated_iter_capture_tokens():
    model = _model()
    batches = list(_loader())
    requests = _requests()
    whole = capture_tokens(model, batches, requests, "cpu")
    concatenated = {key: [] for key in requests}
    for batch_values in iter_capture_tokens(model, batches, requests, "cpu"):
        for key, tensor in batch_values.items():
            concatenated[key].append(tensor)
    assert set(whole) == set(concatenated)
    for key in requests:
        assert len(whole[key]) == len(concatenated[key]) == len(batches)
        for a, b in zip(whole[key], concatenated[key], strict=True):
            assert torch.equal(a, b)


def test_model_device_and_train_mode_restored_after_full_iteration():
    model = _model().train()
    batches = list(_loader())
    for _ in iter_capture_tokens(model, batches, _requests(), "cpu"):
        pass
    assert model.training
    assert next(model.parameters()).device == torch.device("cpu")


def test_model_device_and_train_mode_restored_after_early_close():
    model = _model().train()
    batches = list(_loader())
    gen = iter_capture_tokens(model, batches, _requests(), "cpu")
    next(gen)
    gen.close()
    assert model.training
    assert next(model.parameters()).device == torch.device("cpu")


def test_model_device_and_train_mode_restored_after_exception_in_forward():
    model = _model().train()
    bad_batches = [(torch.ones(2, 5, 8), torch.zeros(2))]
    gen = iter_capture_tokens(model, bad_batches, {"out": (1, "boundary")}, "cpu")
    with pytest.raises(RuntimeError):
        next(gen)
    assert model.training
    assert next(model.parameters()).device == torch.device("cpu")
    assert not model.visual.transformer.resblocks[1]._forward_hooks


def test_lockstep_generators_over_the_same_model_do_not_cross_fire():
    model = _model()
    batches = list(_loader())
    req_a = {"a": (0, "boundary")}
    req_b = {"b": (1, "block_input")}
    gen_a = iter_capture_tokens(model, batches, req_a, "cpu")
    gen_b = iter_capture_tokens(model, batches, req_b, "cpu")
    collected_a, collected_b = [], []
    for va, vb in zip(gen_a, gen_b, strict=True):
        collected_a.append(va["a"])
        collected_b.append(vb["b"])
    expected_a = capture_tokens(model, batches, req_a, "cpu")["a"]
    expected_b = capture_tokens(model, batches, req_b, "cpu")["b"]
    for a, exp in zip(collected_a, expected_a, strict=True):
        assert torch.equal(a, exp)
    for b, exp in zip(collected_b, expected_b, strict=True):
        assert torch.equal(b, exp)


def test_no_grad_tracked_in_yielded_tensors():
    model = _model()
    batches = list(_loader())
    for batch_values in iter_capture_tokens(model, batches, _requests(), "cpu"):
        for tensor in batch_values.values():
            assert not tensor.requires_grad
            assert tensor.grad_fn is None
