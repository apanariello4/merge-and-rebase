"""How much of a source fine-tuning change is only a RoPE-base change (diagnostic, no fitting).

Case: Qwen2.5-1.5B (rope_theta 1e6) -> Qwen2.5-Math-1.5B (rope_theta 1e4). Ariadne's per-block source change is
``S_ft - S_base`` at block outputs on the run's calibration texts, each model under its own config. That change mixes
the Math training with the RoPE-base change. Both models are run under both rope_theta values on the SAME rendered
calibration corpus (resolved by the run's own config through ``resolve_calibration_texts``; its rendered-text sha256
must equal the reference summary's), and per block (residual-stream output of decoder layer j, valid tokens only):

- ``transported`` T = M(own) - B(own)        (what the run transports; own = each model's native rope_theta)
- ``rope_only``   R = B(theta_ft) - B(own)   (base model, only rope_theta changed)
- ``ft_matched``  F = M(own) - B(theta_ft)   (fine-tuning change at the fine-tuned model's rope_theta)
- ``ft_at_base``  G = M(theta_base) - B(own) (fine-tuning change at the base's rope_theta: the pre-368c2ac behaviour)

T = R + F exactly. Reported: Frobenius norms (relative to ||B(own)||), cos(T, R), cos(T, F), cos(R, F), the share of
||T||^2 along R (cos^2), each over all valid tokens and excluding the first token of every sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def _git_state(repo: Path) -> dict[str, Any]:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    ).stdout.strip()
    return {"commit": head, "dirty": bool(dirty)}


def set_rope_theta(cfg: Any, theta: float) -> None:
    """Set rope_theta in both the transformers 4.x (attribute) and 5.x (``rope_parameters``) layouts."""
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict):
        params["rope_theta"] = float(theta)
    if hasattr(cfg, "rope_theta") or not isinstance(params, dict):
        cfg.rope_theta = float(theta)


def native_rope_theta(cfg: Any) -> float:
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return float(cfg.rope_theta)


def load_model(name: str, theta: float, device: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(name)
    set_rope_theta(cfg, theta)
    model = AutoModelForCausalLM.from_pretrained(name, config=cfg, dtype=dtype).to(device).eval()
    inv = model.model.rotary_emb.inv_freq.double().cpu()
    dim = 2 * inv.numel()
    expected = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if not torch.allclose(inv, expected, rtol=1e-5, atol=0.0):
        raise ValueError(f"{name}: inv_freq does not match rope_theta={theta}")
    return model


@torch.no_grad()
def block_states(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], device: str) -> list[torch.Tensor]:
    """Per decoder layer: outputs at valid tokens, rows ordered (sequence, position); plus a first-token row mask."""
    layers = model.model.layers
    store: dict[int, list[torch.Tensor]] = {j: [] for j in range(len(layers))}
    handles = []
    current: dict[str, torch.Tensor] = {}

    def hook(j: int):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            store[j].append(t[current["mask"]].float().cpu())

        return fn

    for j, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(j)))
    try:
        for b in batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            current["mask"] = mask.bool()
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(store[j]) for j in range(len(layers))]


def first_token_rows(batches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    flags = []
    for b in batches:
        m = b["attention_mask"].bool()
        first = torch.zeros_like(m)
        first[torch.arange(m.shape[0]), m.long().argmax(dim=1)] = True
        flags.append(first[m])
    return torch.cat(flags)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double().flatten(), b.double().flatten()
    return float((a64 @ b64) / (a64.norm() * b64.norm()).clamp_min(1e-300))


def block_record(b_own, b_ft, m_own, m_base) -> dict[str, float]:
    t, r, f, g = m_own - b_own, b_ft - b_own, m_own - b_ft, m_base - b_own
    ref = float(b_own.double().norm())
    ct = _cos(t, r)
    return {
        "norm_base": ref,
        "rel_transported": float(t.double().norm()) / ref,
        "rel_rope_only": float(r.double().norm()) / ref,
        "rel_ft_matched": float(f.double().norm()) / ref,
        "rel_ft_at_base": float(g.double().norm()) / ref,
        "cos_transported_rope": ct,
        "cos_transported_ft_matched": _cos(t, f),
        "cos_rope_ft_matched": _cos(r, f),
        "share_transported_along_rope": ct * ct,
        "identity_max_abs": float((t - r - f).abs().max()),
    }


def run(cfg: dict[str, Any], device: str, config_bytes: bytes, config_path: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    from merge_and_rebase.data.llm_calibration import build_text_calibration_loader, resolve_calibration_texts

    out_dir = Path(cfg["output_dir"])
    if out_dir.exists():
        raise FileExistsError(out_dir)
    run_cfg = json.loads(Path(cfg["run_config"]).read_text())
    ref_summary = json.loads(Path(cfg["reference_summary"]).read_text())
    base_name = run_cfg["source_model_name_or_path"]
    ft_name = next(iter(run_cfg["tuned_bodies"].values()))
    n_seq = int(run_cfg.get("num_batches", 10)) * int(run_cfg["calibration_batch_size"])
    tok = AutoTokenizer.from_pretrained(base_name)
    resolved = resolve_calibration_texts(
        calibration_dataset=run_cfg["calibration_dataset"],
        harness_tasks=list(run_cfg.get("harness_tasks", [])),
        n_sequences=n_seq,
        seed=int(run_cfg.get("seed", 0)),
        apply_chat_template=bool(run_cfg.get("harness_apply_chat_template", False)),
        system_instruction=run_cfg.get("harness_system_instruction"),
        tokenizer=tok,
    )
    texts_sha = hashlib.sha256(json.dumps(list(resolved.texts)).encode()).hexdigest()
    ref_sha = ref_summary["calibration_provenance"]["rendered_texts_sha256"]
    if texts_sha != ref_sha:
        raise ValueError(f"rendered calibration texts {texts_sha} differ from the reference summary's {ref_sha}")
    loader = build_text_calibration_loader(
        tokenizer=tok,
        texts=resolved.texts,
        batch_size=int(run_cfg["calibration_batch_size"]),
        max_length=int(run_cfg["calibration_max_length"]),
    )
    batches = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in loader]
    theta_base = native_rope_theta(AutoConfig.from_pretrained(base_name))
    theta_ft = native_rope_theta(AutoConfig.from_pretrained(ft_name))
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    states = {}
    for key, name, theta in (
        ("b_own", base_name, theta_base),
        ("b_ft", base_name, theta_ft),
        ("m_own", ft_name, theta_ft),
        ("m_base", ft_name, theta_base),
    ):
        model = load_model(name, theta, device, dtype)
        states[key] = block_states(model, batches, device)
        del model
        torch.cuda.empty_cache()
    first = first_token_rows(batches)
    blocks = []
    for j in range(len(states["b_own"])):
        s = {k: v[j] for k, v in states.items()}
        blocks.append(
            {
                "block": j,
                "all_tokens": block_record(s["b_own"], s["b_ft"], s["m_own"], s["m_base"]),
                "without_first_token": block_record(*(s[k][~first] for k in ("b_own", "b_ft", "m_own", "m_base"))),
            }
        )
    record = {
        "config_path": config_path,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git": _git_state(Path(__file__).resolve().parents[5]),
        "models": {"base": base_name, "fine_tuned": ft_name},
        "rope_theta": {"base": theta_base, "fine_tuned": theta_ft},
        "dtype": str(dtype),
        "calibration": {
            "rendered_texts_sha256": texts_sha,
            "n_sequences": len(resolved.texts),
            "n_rows": int(first.numel()),
            "n_first_token_rows": int(first.sum()),
        },
        "blocks": blocks,
    }
    out_dir.mkdir(parents=True)
    (out_dir / "rope_probe.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out_dir / 'rope_probe.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config_bytes = Path(args.config).read_bytes()
    run(json.loads(config_bytes), args.device, config_bytes, args.config)


if __name__ == "__main__":
    main()
