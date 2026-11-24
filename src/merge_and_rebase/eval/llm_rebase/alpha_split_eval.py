"""Alpha split evaluation (committed 2026-11-24 from snapshot 20261119e_svdfix, llm_alpha_split_eval.py md5 cda9c7ed;
the only change is the ifeval_k_v1 guard below). Run: python -m merge_and_rebase.eval.llm_rebase.alpha_split_eval ...

L1: choose alpha on a selection slice of GSM8K and report it on the disjoint rest (same docs as llm_rebase).

The llm_rebase pipeline scores every alpha on the whole calibration hold-out (1279 docs), so its best alpha
would be selected on the documents it is reported on. Here the hold-out (identical to llm_rebase's:
resolve_calibration_texts with the run's calibration settings) is split by a seeded permutation into
SEL selection docs and the remaining test docs. For each alpha in the grid, target_base + alpha * tau is
scored on the selection docs; the best alpha (ties -> smaller alpha) is scored once on the test docs.

tau is either a saved transported vector (.pt, from save_transported_tvs_dir) or, with --naive, the
untransported task vector finetuned - source_base (MergeBench minus its base), applied to the target as is.

    python alpha_split_eval.py OUT.json --tv PATH.pt | --naive FT_ID SOURCE_ID | (neither) [--grid 0,0.1,...]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_and_rebase.data.llm_calibration import resolve_calibration_texts
from merge_and_rebase.eval.llm_rebase.harness import run as run_harness

SEED = 0


def holdout_split(task="gsm8k", n_calib=10 * 4, n_sel=300):
    calib = resolve_calibration_texts(calibration_split="val", harness_tasks=[task], n_sequences=n_calib, seed=0)
    docs = list(calib.eval_samples[task])
    perm = torch.randperm(len(docs), generator=torch.Generator().manual_seed(SEED)).tolist()
    sel = sorted(docs[i] for i in perm[:n_sel])
    test = sorted(docs[i] for i in perm[n_sel:])
    return sel, test


def check_ifeval_k_v1(task, n_calib, n_sel, sel, test):
    """IFEval with 40 / 150: the slices must be the committed split ``ifeval_k_v1`` (harness_split.py, sha256-checked),
    so every machine selects alpha on the same 150 prompts and reports on the same 351."""
    if task != "ifeval" or (n_calib, n_sel) != (40, 150):
        return
    from merge_and_rebase.eval.llm_rebase.harness_split import load_harness_split

    ref = load_harness_split("ifeval_k_v1")
    if sorted(sel) != sorted(ref["selection"]) or sorted(test) != sorted(ref["test"]):
        raise RuntimeError("IFEval selection/test slices differ from ifeval_k_v1")


def naive_tv(ft_id, src_id):
    ft = AutoModelForCausalLM.from_pretrained(ft_id, dtype=torch.bfloat16).state_dict()
    src = AutoModelForCausalLM.from_pretrained(src_id, dtype=torch.bfloat16).state_dict()
    return {k: ft[k].float() - src[k].float() for k in ft if k in src and ft[k].shape == src[k].shape}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tv")
    p.add_argument("--naive", nargs=2)
    p.add_argument("--grid", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    p.add_argument("--target", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--fewshot", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-calib", type=int, default=40, help="num_batches * calibration_batch_size of the run")
    p.add_argument("--n-sel", type=int, default=300)
    # gsm8k: flexible-extract (the MergeBench math model scores 0.0 strict-match: no "#### N" answer format
    # but 56.8 flexible, so strict-match would select against the transferred skill).
    p.add_argument("--select", default="flexible")
    p.add_argument("--test-alphas", default="", help="comma list of fixed alphas also scored on the test docs")
    p.add_argument("--key-filter", default="", help="regex: keep only the tau entries whose key matches")
    a = p.parse_args()
    TARGET, TASK, FEWSHOT, BATCH = a.target, a.task, a.fewshot, a.batch
    sel, test = holdout_split(TASK, a.n_calib, a.n_sel)
    check_ifeval_k_v1(TASK, a.n_calib, a.n_sel, sel, test)
    # neither --tv nor --naive: tau = {} scores the target as is (a reference model on the same test docs; grid "0")
    tau = torch.load(a.tv, map_location="cpu") if a.tv else naive_tv(*a.naive) if a.naive else {}
    if a.key_filter:  # e.g. "mlp.down_proj": the same-object ceiling of a method that transports only those weights
        import re
        tau = {k: v for k, v in tau.items() if re.search(a.key_filter, k)}
        print(f"key filter {a.key_filter!r}: {len(tau)} tensors", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to("cuda").eval()
    # missing_bias: materialize (the run's setting): Llama's down_proj has no bias, the transported vector
    # carries one; it is added as a zero bias, so alpha = 0 is still the unmodified target.
    for k in [k for k in tau if k.endswith(".bias") and k not in model.state_dict()]:
        module = model.get_submodule(k[: -len(".bias")])
        module.bias = torch.nn.Parameter(torch.zeros(module.out_features, dtype=module.weight.dtype,
                                                     device=module.weight.device), requires_grad=False)
    base = {k: v.detach().cpu().float() for k, v in model.state_dict().items() if k in tau}
    missing = sorted(set(tau) - set(base))
    if missing:
        raise KeyError(f"tau keys not in the target ({len(missing)}): {missing[:3]}")

    def load(alpha):
        with torch.no_grad():
            sd = model.state_dict()
            for k in tau:
                sd[k].copy_((base[k] + alpha * tau[k]).to(sd[k].dtype))

    def score(alpha, docs):
        load(alpha)
        r = run_harness(tasks=[TASK], model=model, tokenizer=tok, device="cuda", num_fewshot=FEWSHOT,
                        batch_size=BATCH, samples={TASK: docs})
        # IFEval: also the mean of instruction- and prompt-level strict accuracy (the ARIADNE "IFEval Avg."),
        # selectable with --select ifeval_avg_strict
        ks = [k for k in r if k.endswith(("inst_level_strict_acc", "prompt_level_strict_acc"))]
        if len(ks) == 2:
            r["ifeval_avg_strict"] = (r[ks[0]] + r[ks[1]]) / 2
        return r

    curve = {}
    for alpha in [float(x) for x in a.grid.split(",")]:
        curve[alpha] = score(alpha, sel)
        print(f"[sel] alpha {alpha:.2f} {curve[alpha]}", flush=True)
    keys = list(curve[min(curve)])
    key = next((k for k in keys if a.select in k), keys[0])
    print(f"select on {key} (from {keys})", flush=True)
    best = max(sorted(curve), key=lambda al: curve[al][key])
    test_res = score(best, test)
    print(f"[test] alpha {best:.2f} {test_res}", flush=True)
    # extra fixed alphas reported on the test docs (not used for selection), e.g. alpha = 1
    extra = {}
    for alpha in [float(x) for x in a.test_alphas.split(",") if x]:
        if abs(alpha - best) > 1e-9:
            extra[str(alpha)] = score(alpha, test)
            print(f"[test-fixed] alpha {alpha:.2f} {extra[str(alpha)]}", flush=True)
    json.dump({"tv": a.tv, "naive": a.naive, "target": TARGET, "n_sel": len(sel), "n_test": len(test),
               "select_metric": key, "curve_sel": {str(k): v for k, v in curve.items()}, "best_alpha": best,
               "test": test_res, "test_fixed_alpha": extra, "tau_keys": len(tau)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
