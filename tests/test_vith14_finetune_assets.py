from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/merge_and_rebase/finetune/configs/vision-ft-vith14-laion.yaml"
SMOKE_CONFIG = ROOT / "src/merge_and_rebase/finetune/configs/vision-ft-vith14-laion-smoke.yaml"
LAUNCHER = ROOT / "scripts/run_vision_finetune_vith14.slurm"


TASKS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
}


def test_vith14_full_config_matches_requested_protocol():
    payload = yaml.safe_load(CONFIG.read_text())
    common = payload["common"]
    assert common["backbone"] == {
        "name": "openclip",
        "clip_model": "ViT-H-14",
        "clip_pretrained": "laion2b_s32b_b79k",
    }
    assert common["dtype"] == "fp32"
    assert common["data"]["batch_size"] == 4
    assert common["train"]["accumulate_grad_batches"] == 32
    assert common["train"]["optimizer"]["name"] == "adamw"
    assert common["train"]["lr"] == 1.0e-5
    assert common["train"]["weight_decay"] == 0.1
    assert common["train"]["lr_scheduler"]["name"] == "cosine"
    assert common["train"]["grad_clip_norm"] == 1.0
    assert common["train"]["early_stopping"] is False
    assert payload["datasets_order"] == TASKS
    assert {task: payload["datasets"][task]["train"]["epochs"] for task in TASKS} == EPOCHS


def test_smoke_config_isolated_and_short():
    payload = yaml.safe_load(SMOKE_CONFIG.read_text())
    assert payload["common"]["backbone"]["clip_model"] == "ViT-H-14"
    assert payload["common"]["backbone"]["clip_pretrained"] == "laion2b_s32b_b79k"
    assert payload["common"]["output"]["out_dir"].endswith("finetune_smoke")
    assert payload["common"]["train"]["max_train_batches"] == 1
    assert payload["datasets_order"] == ["MNIST"]
    assert payload["datasets"]["MNIST"]["train"]["epochs"] == 1


def test_slurm_launcher_has_eight_tasks_and_cache_safety():
    text = LAUNCHER.read_text()
    assert "#SBATCH --array=0-7" in text
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "export HF_DATASETS_CACHE=\"${HF_HOME}/datasets\"" in text
    assert "HF_DATASETS_OFFLINE" not in text
    assert "--skip-existing-task-vectors" in text
    for task in TASKS:
        assert task in text
