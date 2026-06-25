from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    tasks: Sequence[str]
    # task -> (hf_path, hf_config, split_map)
    resolver: Any


VISION8_TASKS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
VISION14_TASKS = VISION8_TASKS + ["CIFAR100", "STL10", "Flowers102", "OxfordIIITPet", "PCAM", "FER2013"]
VISION20_TASKS = VISION14_TASKS + ["EMNIST", "CIFAR10", "Food101", "FashionMNIST", "RenderedSST2", "KMNIST"]
VISION_IMAGENET_TASKS = ["ImageNet1K", "ImageNet21KP"]
VISION_SUPPORTED_TASKS = VISION20_TASKS + VISION_IMAGENET_TASKS


def _vision_spec(task: str) -> tuple[str, str | None, dict[str, str]]:
    mapping = {
        "SUN397": ("tanganke/sun397", None, {"train": "train", "test": "test"}),
        "Cars": ("tanganke/stanford_cars", None, {"train": "train", "test": "test"}),
        "RESISC45": ("tanganke/resisc45", None, {"train": "train", "test": "test"}),
        "EuroSAT": ("tanganke/eurosat", None, {"train": "train", "test": "test"}),
        "GTSRB": ("tanganke/gtsrb", None, {"train": "train", "test": "test"}),
        "MNIST": ("ylecun/mnist", None, {"train": "train", "test": "test"}),
        "DTD": ("tanganke/dtd", None, {"train": "train", "test": "test"}),
        "CIFAR100": ("tanganke/cifar100", None, {"train": "train", "test": "test"}),
        "STL10": ("tanganke/stl10", None, {"train": "train", "test": "test"}),
        "Flowers102": ("dpdl-benchmark/oxford_flowers102", None, {"train": "train", "test": "test"}),
        "OxfordIIITPet": ("timm/oxford-iiit-pet", None, {"train": "train", "test": "test"}),
        "PCAM": ("1aurent/PatchCamelyon", None, {"train": "train", "test": "test"}),
        "FER2013": ("clip-benchmark/wds_fer2013", None, {"train": "train", "test": "test"}),
        "EMNIST": ("tanganke/emnist_mnist", None, {"train": "train", "test": "test"}),
        "CIFAR10": ("tanganke/cifar10", None, {"train": "train", "test": "test"}),
        "Food101": ("ethz/food101", None, {"train": "train", "test": "validation"}),
        "FashionMNIST": ("zalando-datasets/fashion_mnist", None, {"train": "train", "test": "test"}),
        "RenderedSST2": ("nateraw/rendered-sst2", None, {"train": "train", "test": "test"}),
        "KMNIST": ("tanganke/kmnist", None, {"train": "train", "test": "test"}),
        "SVHN": ("ufldl-stanford/svhn", "cropped_digits", {"train": "train", "test": "test"}),
        "ImageNet1K": ("ILSVRC/imagenet-1k", None, {"train": "train", "test": "validation"}),
        "ImageNet21KP": ("timm/imagenet-w21-p", None, {"train": "train", "test": "validation"}),
    }
    if task not in mapping:
        raise ValueError(f"Unknown vision task '{task}'. Choose from {VISION_SUPPORTED_TASKS}")
    return mapping[task]


SUITES: dict[str, SuiteSpec] = {
    "vision8": SuiteSpec(name="vision8", tasks=VISION8_TASKS, resolver=_vision_spec),
    "vision14": SuiteSpec(name="vision14", tasks=VISION14_TASKS, resolver=_vision_spec),
    "vision20": SuiteSpec(name="vision20", tasks=VISION20_TASKS, resolver=_vision_spec),
}
