"""Regenerate (or --check) src/merge_and_rebase/eval/llm_rebase/ifeval_k_v1.json, the fixed IFEval selection/test split.

    python -m merge_and_rebase.eval.llm_rebase.make_ifeval_split          # write the file
    python -m merge_and_rebase.eval.llm_rebase.make_ifeval_split --check  # exit 1 if the stored file differs from a regeneration
"""

import argparse
import json
import sys

from merge_and_rebase.eval.llm_rebase.harness_split import SPLITS, build_split_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = SPLITS["ifeval_k_v1"]
    text = json.dumps(build_split_file("ifeval_k_v1"), indent=1) + "\n"
    if args.check:
        same = path.exists() and path.read_text() == text
        print(f"{path}: {'identical' if same else 'DIFFERS'}")
        return 0 if same else 1
    path.write_text(text)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
