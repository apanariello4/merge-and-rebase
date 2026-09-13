from types import SimpleNamespace

import pytest
import torch
from torch import nn

from merge_and_rebase.eval import vision_brace_source_merge as source_merge
from merge_and_rebase.eval.vision_brace_source_merge import arm_vector, tensor_stats


def _v(x, y):
    return {"visual.a": torch.tensor([x, y], dtype=torch.float32), "visual.b": torch.tensor([x + 1], dtype=torch.float32)}


def test_stats_are_double_precision_and_inputs_are_unchanged():
    a, b = _v(3, 4), _v(4, 3)
    before = {k: v.clone() for k, v in a.items()}
    stats = tensor_stats(a, b)
    assert stats["norm_a"] == pytest.approx((41**0.5))
    assert stats["norm_b"] == pytest.approx((50**0.5))
    assert torch.equal(a["visual.a"], before["visual.a"])


def test_norm_control_rejects_zero_vector():
    with pytest.raises(ValueError, match="Zero vector"):
        arm_vector({"visual.a": torch.zeros(2), "visual.b": torch.zeros(1)}, _v(1, 1), "independent_shared_norm")


def test_arms_preserve_key_shapes_and_midpoint():
    i, s = _v(1, 2), _v(3, 4)
    out = arm_vector(i, s, "midpoint_raw")
    assert torch.equal(out["visual.a"], torch.tensor([2., 3.]))
    assert set(out) == set(i)
    assert all(out[k].shape == i[k].shape for k in i)


def test_source_template_calls_keyword_only_block_extension(monkeypatch):
    calls = []
    def fake_extension(*, source_base_model, source_ft_model, **kwargs):
        calls.append({"source_base_model": source_base_model, "source_ft_model": source_ft_model, **kwargs})
    monkeypatch.setattr(source_merge, "run_block_extension", fake_extension)
    template = source_merge._template(SimpleNamespace(model=nn.Linear(2, 2)), 2, "cpu")
    assert isinstance(template, nn.Module)
    assert calls and {"source_base_model", "source_ft_model", "calibration_loader", "target_layers_total", "config", "device"} <= set(calls[0])


def test_midpoint_shared_norm_uses_raw_midpoint_direction():
    independent = {"visual.a": torch.tensor([2.0, 0.0])}
    shared = {"visual.a": torch.tensor([0.0, 1.0])}
    result = arm_vector(independent, shared, "midpoint_shared_norm")
    expected = torch.tensor([1.0, 0.5]) / torch.tensor(1.25).sqrt()
    assert torch.allclose(result["visual.a"], expected)
    assert tensor_stats(result, result)["norm_a"] == pytest.approx(1.0)


def test_alpha_grid_is_exact_sorted_unique_and_finite():
    assert source_merge._alphas({"alpha_values": [0, 0.01, 1]}) == [0.0, 0.01, 1.0]
    for bad in ([0, 1, 1], [0, 1, float("nan")], [-0.1, 0, 1], [1, 0]):
        with pytest.raises(ValueError):
            source_merge._alphas({"alpha_values": bad})


def test_run_selects_on_validation_then_uses_same_alpha_for_merged_and_isolated(monkeypatch):
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([0.0]))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()

    calls = []

    class Classifier:
        def __init__(self, model=None, **kwargs):
            self.model = model or Model()
            self.tokenizer = None
            self.preprocess = None
            self.normalize = True
            self.logit_scale = 1.0
            self._zs_text_features = torch.empty(0)

        @classmethod
        def build(cls, cfg):
            return cls()

        def build_zeroshot_text_features(self, classnames, cfg, **kwargs):
            self._zs_text_features = torch.tensor([float(classnames[0])])

        def top1(self, loader, device):
            split, task = loader
            calls.append(split)
            weight = float(self.model.visual.w.detach().item())
            # Validation peaks at merged weight 1.0, so alpha=.5 for two +1 TVs.
            return max(0.0, 1.0 - abs(weight - (1.0 if split == "val" else 0.75)))

    base = {"visual.w": torch.tensor([0.0])}
    metadata = {"base_sha256": "same"}
    bank = {
        task: {"base": base, "ft": {"visual.w": torch.tensor([1.0])},
               "tv": {"visual.w": torch.tensor([1.0])}, "metadata": metadata}
        for task in ("Cars", "EuroSAT")
    }
    manifests = {"independent": {"target_depth": 24}, "shared": {"target_depth": 24}}
    monkeypatch.setattr(source_merge, "_validate_banks", lambda cfg, tasks: (bank, bank, manifests))
    monkeypatch.setattr(source_merge, "OpenClipClassifier", Classifier)
    monkeypatch.setattr(source_merge, "_template", lambda clf, depth, device: clf.model)
    monkeypatch.setattr(
        source_merge,
        "_task_loader_context",
        lambda task, **kwargs: (
            type("Loaders", (), {"val": ("val", task), "test": ("test", task)})(),
            ["1" if task == "Cars" else "2"],
            object(),
        ),
    )
    monkeypatch.setattr(source_merge, "resolve_eval_split_loader", lambda loaders, split: getattr(loaders, split))
    monkeypatch.setattr(source_merge, "load_into_model", lambda model, state, strict=False: model.load_state_dict(state, strict=False))

    result = source_merge.run({
        "suite": "vision8", "tasks": "Cars,EuroSAT", "arm": "shared",
        "recipient_base": "Cars", "alpha_values": [0.0, 0.5, 1.0],
        "device": "cpu", "source_clip_model": "ViT-B-16",
        "source_clip_pretrained": "datacomp_xl_s13b_b90k",
    })
    assert result["selected_alpha"] == 0.5
    assert calls.index("test") == 6  # 3 validation points x 2 tasks first.
    for task in ("Cars", "EuroSAT"):
        selected = result["test"]["shared"][task]["selected"]
        assert selected["merged"] == pytest.approx(0.75)
        assert selected["isolated"] == pytest.approx(0.75)
        assert selected["merged_minus_isolated"] == pytest.approx(0.0)
    assert result["transport_method"] == "none"
    assert result["recipient_base_sha256"] == source_merge.state_dict_sha256(base)
