"""Self-generated IFEval-style calibration prompts: Qwen2.5-0.5B-Instruct (the source fine-tuned model) writes user
requests with checkable formatting constraints. The meta-instruction names only the public IFEval instruction *types*;
no IFEval prompt is shown to the model. Output: {"source", "meta_prompt", "gen", "prompts"} JSON.

usage: python gen_self_prompts.py OUT.json [--n 400] [--seed 0] [--model Qwen/Qwen2.5-0.5B-Instruct]
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

META = (
    # v2 (v1 listed "output in JSON" and the 0.5B model mostly wrapped its request in JSON; v1 kept in self_prompts_raw.json)
    "Write one request that a user might send to an AI assistant, asking it to write something (for example a story, "
    "an email, an essay, a poem, a list, an explanation or a review) on any topic you choose. The request must include "
    "one or two explicit, checkable formatting constraints, for example: a minimum or maximum number of words or "
    "sentences, an exact number of paragraphs or bullet points, all lowercase or all capital letters, no commas, "
    "keywords that must appear, a title in double angular brackets, a postscript starting with P.S., "
    "or an exact phrase to end with. Write the request as plain text, starting with a verb such as Write, Describe, "
    "Explain or List. Output only the request itself, nothing else."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    text = tok.apply_chat_template([{"role": "user", "content": META}], tokenize=False, add_generation_prompt=True)
    gen = dict(do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=160)
    outs = []
    while len(outs) < a.n:
        enc = tok([text] * a.batch, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**enc, **gen, pad_token_id=tok.eos_token_id)
        for row in ids[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip().strip('"').strip())
    raw = outs[: a.n]
    seen, keep = set(), []
    for s in raw:  # dedup and length filter; nothing else
        k = " ".join(s.lower().split())
        if 40 <= len(s) <= 700 and k not in seen and not s.lstrip().startswith(("{", "[", "`")):
            seen.add(k)
            keep.append(s)
    json.dump({"source": a.model, "meta_prompt": META, "gen": gen, "seed": a.seed, "n_raw": len(raw),
               "n_kept": len(keep), "raw": raw, "prompts": keep}, open(a.out, "w"), indent=1)
    print(f"kept {len(keep)} of {len(raw)}")


if __name__ == "__main__":
    main()
