"""``load_ckpt(mmap=True)`` (checkpoint-base classification) returns the same state dict as a full read."""

from __future__ import annotations

import torch

from merge_and_rebase.io.ckpt import load_ckpt


def test_mmap_load_matches_full_load(tmp_path):
    sd = {"visual.proj": torch.randn(4, 3), "visual.ln_post.weight": torch.randn(4)}
    for name, legacy in (("zip.pt", False), ("legacy.pt", True)):
        path = tmp_path / name
        torch.save({"state_dict": sd}, path, _use_new_zipfile_serialization=not legacy)
        full, mapped = load_ckpt(str(path)), load_ckpt(str(path), mmap=True)
        assert full.keys() == mapped.keys()
        assert all(torch.equal(full[k], mapped[k]) for k in full)
