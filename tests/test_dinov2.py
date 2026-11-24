import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

from merge_and_rebase.eval._dinov2_merge import prepare_merge, validate_checkpoint
from merge_and_rebase.finetune._dinov2 import cpu_state, train_epochs, validate_options
from merge_and_rebase.merge.registry import list_methods
from merge_and_rebase.models.dinov2_classifier import DinoPreprocess, DinoV2Classifier


@pytest.fixture
def model():
    torch.set_num_threads(1)
    return DinoV2Classifier(
        Dinov2Model(
            Dinov2Config(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                image_size=28,
                patch_size=14,
            )
        ),
        3,
    )


@pytest.mark.parametrize("mode", ["full", "linear_probe"])
def test_training_gradients(model, mode):
    model.set_training_mode(mode)
    model.train()
    x = torch.randn(2, 3, 28, 28)
    loss = model(x).square().mean()
    loss.backward()
    assert model.head.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters()) == (mode == "full")
    assert model.backbone.training == (mode == "full")
    assert model(x).shape == (2, 3)


def test_grayscale_processor():
    processor = BitImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    x = DinoPreprocess(processor)(Image.new("L", (32, 32)))
    assert x.shape == (3, 28, 28)


def test_checkpoint_roundtrip(model, tmp_path):
    payload = {
        "model_config": model.backbone.config.to_dict(),
        "backbone_state": cpu_state(model.backbone),
        "head_state": cpu_state(model.head),
        "classnames": ["a", "b", "c"],
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    restored = DinoV2Classifier.from_checkpoint(torch.load(path, weights_only=True)).eval()
    model.eval()
    x = torch.randn(2, 3, 28, 28)
    torch.testing.assert_close(restored(x), model(x))
    merged = {**payload, "tasks": {"task": {"head_state": payload["head_state"], "classnames": payload["classnames"]}}}
    restored = DinoV2Classifier.from_checkpoint(merged, "task").eval()
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("method", list_methods())
def test_all_dense_methods(model, method):
    base = cpu_state(model.backbone)
    torch.manual_seed(42)
    tuned = [{k: v + torch.randn_like(v) * 0.01 for k, v in base.items()} for _ in range(2)]
    head = cpu_state(model.head)
    merger, prepared = prepare_merge(base, tuned, method)
    zero = merger.apply(prepared, alpha=0.0)
    merged = {**base, **merger.apply(prepared, alpha=0.5)}
    assert merged.keys() == base.keys()
    for k in base:
        torch.testing.assert_close(zero.get(k, base[k]), base[k])
        assert torch.isfinite(merged[k]).all()
    for k, v in head.items():
        torch.testing.assert_close(v, model.head.state_dict()[k])


def test_arithmetic_and_average(model):
    base = cpu_state(model.backbone)
    tuned = [{k: v + offset for k, v in base.items()} for offset in (1.0, 3.0)]
    for method, expected in [("task_arithmetic", 4.0), ("weighted_average", 2.0)]:
        merger, prepared = prepare_merge(base, tuned, method)
        result = merger.apply(prepared, alpha=1.0)
        for k in result:
            if not torch.equal(result[k], base[k]):
                torch.testing.assert_close(result[k], base[k] + expected)
    merger, prepared = prepare_merge(base, [base, base], "task_arithmetic")
    for k, value in merger.apply(prepared, alpha=1.0).items():
        torch.testing.assert_close(value, base[k])


@pytest.mark.parametrize("mode", ["full", "linear_probe"])
def test_training_loop(model, mode):
    batch = (torch.randn(2, 3, 28, 28), torch.tensor([0, 1]))
    loaders = SimpleNamespace(train=[batch] * 3, val=[batch], test=[batch])
    original = cpu_state(model.backbone)
    cfg = {"strategy": {"name": mode}, "train": {"epochs": 2, "lr": 0.01, "accumulate_grad_batches": 2}}
    best, last, summary = train_epochs(model, loaders, cfg, torch.device("cpu"))
    assert best.keys() == last.keys()
    assert 0 <= summary["test_top1"] <= 1
    if mode == "linear_probe":
        for k, value in original.items():
            torch.testing.assert_close(value, model.backbone.state_dict()[k])


def test_base_validation(model, tmp_path):
    base = {"identity": {"model_id": "test"}, "backbone_state": cpu_state(model.backbone)}
    path = tmp_path / "base.pt"
    torch.save(base, path)
    tuned = {**deepcopy(base), "format": "dinov2_classifier_v1", "task": "MNIST", "base_checkpoint": str(path)}
    validate_checkpoint(base, tuned, "MNIST")
    tuned["identity"]["model_id"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint(base, tuned, "MNIST")


def test_reject_clip_options():
    with pytest.raises(ValueError, match="text_prompt_tuning"):
        validate_options({"strategy": {"text_prompt_tuning": {"enabled": True}}})


def test_training_and_merge_entrypoints(model, tmp_path, monkeypatch):
    import json
    import sys

    import yaml

    from merge_and_rebase.eval import _dinov2_merge, vision_merge
    from merge_and_rebase.finetune import _dinov2, train_vision

    batch = (torch.randn(2, 3, 28, 28), torch.tensor([0, 1]))
    loaders = SimpleNamespace(train=[batch], val=[batch], test=[batch], classnames=["a", "b", "c"])
    processor = BitImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    monkeypatch.setattr(DinoV2Classifier, "pretrained", lambda *args: (deepcopy(model.backbone), processor))
    monkeypatch.setattr(_dinov2, "make_loaders", lambda *args: loaders)
    monkeypatch.setattr(_dinov2_merge, "make_loaders", lambda *args: loaders)
    for mode in ("full", "linear_probe", "classifier_then_backbone"):
        out = tmp_path / mode
        config = {
            "common": {
                "backbone": {"name": "dinov2", "model_id": "test"},
                "device": "cpu",
                "strategy": {"name": mode},
                "train": {"epochs": 1},
                "data": {"num_workers": 0},
                "output": {"out_dir": str(out), "save_last_epoch": True},
                "logging": {"use_wandb": False},
            },
            "datasets_order": ["MNIST", "SVHN"],
        }
        path = tmp_path / "train.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys, "argv", ["train_vision", "--vision-config", str(path)])
        train_vision.main()
        assert (out / "MNIST" / f"{mode}_last.pt").exists()
        if mode == "classifier_then_backbone":
            assert (out / "MNIST" / f"{mode}_classifier_best.pt").exists()
            first = torch.load(out / "MNIST" / f"{mode}_classifier_best.pt", weights_only=True)
            second = torch.load(out / "MNIST" / f"{mode}_best.pt", weights_only=True)
            for key in first["head_state"]:
                assert torch.equal(first["head_state"][key], second["head_state"][key])
        cfg = {
            "backbone": {"name": "dinov2"},
            "tasks": "MNIST,SVHN",
            "device": "cpu",
            "tuned_ckpts": {task: str(out / task / f"{mode}_best.pt") for task in ["MNIST", "SVHN"]},
            "alphas": [0.0, 0.5],
            "save_merged": str(out / "merged.pt"),
            "logging": {"use_wandb": False},
        }
        path = tmp_path / "merge.json"
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr(sys, "argv", ["vision_merge", "--config", str(path)])
        vision_merge.main()
        payload = torch.load(out / "merged.pt", weights_only=True)
        restored = DinoV2Classifier.from_checkpoint(payload, "MNIST")
        assert restored(batch[0]).shape == (2, 3)
        for task in ["MNIST", "SVHN"]:
            tuned = torch.load(cfg["tuned_ckpts"][task], weights_only=True)
            for key, value in payload["tasks"][task]["head_state"].items():
                torch.testing.assert_close(value, tuned["head_state"][key])


def test_class_order_mismatch(model, tmp_path, monkeypatch):
    from merge_and_rebase.eval import _dinov2_merge

    processor = BitImageProcessor()
    identity = {
        "model_id": "test",
        "processor": json.loads(processor.to_json_string()),
        "model_config": model.backbone.config.to_dict(),
    }
    base_path = tmp_path / "base.pt"
    torch.save({"identity": identity, "backbone_state": cpu_state(model.backbone)}, base_path)
    tuned = {
        "format": "dinov2_classifier_v1",
        "task": "MNIST",
        "identity": identity,
        "base_checkpoint": str(base_path),
        "backbone_state": cpu_state(model.backbone),
        "head_state": cpu_state(model.head),
        "classnames": ["a", "b", "c"],
        "data": {},
        "seed": 42,
    }
    path = tmp_path / "task.pt"
    torch.save(tuned, path)
    monkeypatch.setattr(_dinov2_merge, "make_loaders", lambda *args: SimpleNamespace(classnames=["c", "b", "a"]))
    with pytest.raises(ValueError, match="Class order"):
        _dinov2_merge.run_merge({"tasks": "MNIST", "tuned_ckpts": {"MNIST": str(path)}})


def test_backbone_only_keeps_head_frozen(model):
    head = cpu_state(model.head)
    backbone = cpu_state(model.backbone)
    model.set_training_mode("backbone_only")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 28, 28)), torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.head.parameters())
    assert any(p.grad is not None for p in model.backbone.parameters())
    optimizer.step()
    for key, value in head.items():
        assert torch.equal(model.head.state_dict()[key], value)
    assert any(not torch.equal(value, model.backbone.state_dict()[key]) for key, value in backbone.items())


def test_classifier_then_backbone_stages(model):
    from merge_and_rebase.finetune._dinov2 import train_variant

    batch = (torch.randn(2, 3, 28, 28), torch.tensor([0, 1]))
    loaders = SimpleNamespace(train=[batch] * 3, val=[batch], test=[batch])
    original_backbone = cpu_state(model.backbone)
    original_head = cpu_state(model.head)
    stage_states = {}
    cfg = {
        "strategy": {"name": "classifier_then_backbone"},
        "train": {"epochs": 2, "lr": 0.01, "accumulate_grad_batches": 2, "classifier_stage": {"epochs": 1, "lr": 0.01}},
    }

    def capture(stage, best, last, metrics):
        stage_states[stage] = best

    _, _, summary = train_variant(model, loaders, cfg, torch.device("cpu"), on_checkpoint=capture)
    for key, value in original_backbone.items():
        assert torch.equal(stage_states["classifier"]["backbone." + key], value)
    assert any(
        not torch.equal(stage_states["classifier"]["head." + key], value) for key, value in original_head.items()
    )
    for key in original_head:
        assert torch.equal(stage_states["classifier"]["head." + key], stage_states["backbone"]["head." + key])
    assert any(
        not torch.equal(stage_states["backbone"]["backbone." + key], value) for key, value in original_backbone.items()
    )
    assert summary["stages"]["classifier"]["last_epoch"] == 0
    assert summary["stages"]["backbone"]["last_epoch"] == 1


@pytest.mark.parametrize("fixed_split", [None, "val", "test"])
def test_merge_selects_normalized_validation_and_preserves_task_heads(model, tmp_path, monkeypatch, fixed_split):
    from merge_and_rebase.eval import _dinov2_merge

    processor = BitImageProcessor()
    identity = {
        "model_id": "test",
        "processor": json.loads(processor.to_json_string()),
        "model_config": model.backbone.config.to_dict(),
    }
    base = cpu_state(model.backbone)
    base_path = tmp_path / "base.pt"
    torch.save({"identity": identity, "backbone_state": base}, base_path)
    paths, summaries = {}, {}
    for task, reference in [("MNIST", 1.0), ("SVHN", 0.2)]:
        payload = {
            "format": "dinov2_classifier_v1",
            "task": task,
            "identity": identity,
            "base_checkpoint": str(base_path),
            "backbone_state": {key: value + 0.01 for key, value in base.items()},
            "head_state": cpu_state(model.head),
            "classnames": ["a", "b", "c"],
            "data": {},
            "seed": 42,
            "model_config": identity["model_config"],
        }
        paths[task] = str(tmp_path / f"{task}.pt")
        torch.save(payload, paths[task])
        summaries[task] = str(tmp_path / f"{task}.json")
        (tmp_path / f"{task}.json").write_text(json.dumps({"best_val_top1": reference, "test_top1": reference}))
    monkeypatch.setattr(
        _dinov2_merge,
        "make_loaders",
        lambda task, *args: SimpleNamespace(classnames=["a", "b", "c"], val=(task, "val"), test=(task, "test")),
    )
    test_calls = []

    def fake_top1(current, loader, device):
        import fcntl

        with (tmp_path / "prepare.lock").open("a") as available:
            fcntl.flock(available.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(available.fileno(), fcntl.LOCK_UN)
        task, split = loader
        if split == "test":
            test_calls.append(task)
        shifted = not torch.equal(
            next(current.backbone.parameters()).detach().cpu(), next(model.backbone.parameters()).detach().cpu()
        )
        return {"MNIST": 0.7, "SVHN": 0.3}[task] if shifted else {"MNIST": 0.9, "SVHN": 0.2}[task]

    monkeypatch.setattr(_dinov2_merge, "top1", fake_top1)
    result_path = tmp_path / "results.json"
    output = tmp_path / "merged.pt"
    _dinov2_merge.run_merge(
        {
            "tasks": "MNIST, SVHN",
            "device": "cpu",
            "tuned_ckpts": paths,
            "single_task_summaries": summaries,
            "exclusive_merge_lock": str(tmp_path / "prepare.lock"),
            "selection_metric": "mean_normalized_top1",
            "alphas": [0.0, 1.0] if fixed_split is None else [1.0],
            "fixed_eval_split": fixed_split,
            "results": str(result_path),
            "save_merged": str(output),
        }
    )
    result = json.loads(result_path.read_text())
    # Alpha 0 has the higher raw average, but alpha 1 has the higher normalized score.
    assert result["alpha"] == 1.0
    assert result[fixed_split or "val"]["mean_normalized_top1"] == pytest.approx(1.1)
    assert test_calls == ([] if fixed_split == "val" else ["MNIST", "SVHN"])
    if fixed_split:
        assert ({"val", "test"} - {fixed_split}).isdisjoint(result)
    assert json.loads(result_path.with_suffix(".progress.json").read_text())["status"] == "completed"
    merged = torch.load(output, weights_only=True)
    for task in paths:
        for key, value in model.head.state_dict().items():
            assert torch.equal(value, merged["tasks"][task]["head_state"][key])


@pytest.mark.parametrize("fail", [False, True])
def test_exclusive_merge_slot_held_and_released(tmp_path, fail):
    import fcntl

    from merge_and_rebase.eval import _dinov2_merge

    path = tmp_path / "shared_iso_tsv.lock"

    def check_slot(cfg):
        # Another independently opened descriptor cannot enter the same slot.
        with path.open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if fail:
            raise RuntimeError("simulated job failure")
        return "finished"

    cfg = {"exclusive_merge_lock": str(path), "method": "isoc_merge"}
    if fail:
        with pytest.raises(RuntimeError, match="simulated"):
            with _dinov2_merge._merge_preparation_slot(cfg):
                check_slot(cfg)
    else:
        with _dinov2_merge._merge_preparation_slot(cfg):
            assert check_slot(cfg) == "finished"
    with path.open("a") as next_job:
        fcntl.flock(next_job.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(next_job.fileno(), fcntl.LOCK_UN)


def test_dinov2_checkpoint_load_resolves_hub_reference(model, tmp_path, monkeypatch):
    from merge_and_rebase.eval import _dinov2_merge

    local = tmp_path / "base.pt"
    torch.save({"backbone_state": cpu_state(model.backbone)}, local)
    reference = "hf-hub:merge-and-rebase/checkpoints/finetune/dinov2-base/facebook/base.pt"
    resolved = []

    def resolve(path):
        resolved.append(path)
        return str(local)

    monkeypatch.setattr(_dinov2_merge, "resolve_ckpt_path", resolve)
    payload = _dinov2_merge.load_checkpoint(reference)
    assert resolved == [reference]
    for key, value in model.backbone.state_dict().items():
        assert torch.equal(value, payload["backbone_state"][key])


@pytest.mark.parametrize("tasks", ["", "MNIST,MNIST", "unknown"])
def test_merge_rejects_invalid_or_duplicate_tasks_before_loading(tasks):
    from merge_and_rebase.eval._dinov2_merge import run_merge

    with pytest.raises(ValueError, match="Invalid tasks"):
        run_merge({"tasks": tasks})


def test_documented_training_presets_cover_their_suites():
    from pathlib import Path

    import yaml

    from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES
    from merge_and_rebase.finetune.train_vision import _deep_update, build_parser, resolve_tasks

    root = Path(__file__).resolve().parents[1] / "configs/examples/dinov2"
    for suite in ("vision8", "vision20"):
        for mode in ("full", "linear_probe", "classifier_then_backbone"):
            path = root / f"{suite}_{mode}.yaml"
            cfg = yaml.safe_load(path.read_text())
            args = build_parser().parse_args(["--vision-config", str(path)])
            assert resolve_tasks(args, cfg) == list(SUITES[suite].tasks)
            for task in SUITES[suite].tasks:
                merged = _deep_update(deepcopy(cfg["common"]), cfg["datasets"][task])
                validate_options(merged)
                assert merged["strategy"]["name"] == mode
                assert merged["train"]["epochs"] > 0
            if suite == "vision20":
                args.suite = "vision14"
                assert resolve_tasks(args, cfg) == list(SUITES["vision14"].tasks)


def test_documented_merge_presets_reference_training_outputs():
    from pathlib import Path

    import yaml

    from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES

    root = Path(__file__).resolve().parents[1] / "configs/examples/dinov2"
    for suite in ("vision8", "vision14", "vision20"):
        for mode in ("full", "classifier_then_backbone"):
            suffix = "merge" if mode == "full" else "classifier_then_backbone_merge"
            cfg = json.loads((root / f"{suite}_{suffix}.json").read_text())
            training_suite = "vision8" if suite == "vision8" else "vision20"
            training = yaml.safe_load((root / f"{training_suite}_{mode}.yaml").read_text())
            directory = Path(training["common"]["output"]["out_dir"])
            tasks = list(SUITES[suite].tasks)
            assert list(cfg["tuned_ckpts"]) == tasks
            assert cfg["base_ckpt"] == str(directory / "base.pt")
            assert cfg["selection_metric"] == "mean_normalized_top1"
            assert (cfg["alpha_min"], cfg["alpha_max"], cfg["alpha_step"]) == (0.0, 2.0, 0.1)
            for task in tasks:
                assert cfg["tuned_ckpts"][task] == str(directory / task / f"{mode}_best.pt")
                assert cfg["single_task_summaries"][task] == str(directory / task / f"{mode}.json")
