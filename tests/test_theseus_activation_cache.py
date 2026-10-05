import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from merge_and_rebase.rebase.methods.theseus import TheseusRebase, _activation_cache_fingerprint


def test_theseus_reuses_activation_cache(tmp_path, monkeypatch) -> None:
    source = nn.Linear(2, 2, bias=False)
    target = nn.Linear(2, 2, bias=False)
    source.register_parameter("scalar", nn.Parameter(torch.tensor(1.0)))
    batches = [(torch.ones(2, 2),)]
    method = TheseusRebase()
    calls = 0

    from merge_and_rebase.rebase.methods import theseus

    original_collect = theseus.collect_activations

    def counted_collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(theseus, "collect_activations", counted_collect)
    kwargs = dict(
        source_model=source,
        target_model=target,
        source_dataloader=batches,
        target_dataloader=batches,
        device="cpu",
        n_batches=1,
        patch_qkv=False,
        verbose=False,
        activation_cache_dir=str(tmp_path),
        activation_cache_mode="auto",
    )
    first = method.prepare(**kwargs)
    second = method.prepare(**kwargs)

    assert calls == 1
    assert first["activation_registry"].keys() == second["activation_registry"].keys()
    assert len(list(tmp_path.glob("theseus_activations_*.pt"))) == 1


def test_activation_cache_fingerprint_includes_whitening_and_dataset_identity():
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)

    class DatasetA(torch.utils.data.Dataset):
        _fingerprint = "dataset-a"

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.zeros(2), 0

    class DatasetB(DatasetA):
        _fingerprint = "dataset-b"

    loader_a = DataLoader(DatasetA(), batch_size=2)
    loader_b = DataLoader(DatasetB(), batch_size=2)

    def fingerprint(loader, whiten_power):
        return _activation_cache_fingerprint(
            source_model=source,
            target_model=target,
            source_dataloader=loader,
            target_dataloader=loader,
            seq_align="interpolate2d",
            n_batches=2,
            seed=0,
            batch_size=2,
            cache_key=None,
            whiten_power=whiten_power,
            whiten_eps=1e-5,
        )

    assert fingerprint(loader_a, 0.0) != fingerprint(loader_a, 0.25)
    assert fingerprint(loader_a, 0.0) != fingerprint(loader_b, 0.0)
