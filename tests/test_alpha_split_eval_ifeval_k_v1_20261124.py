"""alpha_split_eval refuses IFEval slices that differ from the committed split ifeval_k_v1."""

import pytest

from merge_and_rebase.eval.llm_rebase.alpha_split_eval import check_ifeval_k_v1
from merge_and_rebase.eval.llm_rebase.harness_split import load_harness_split


def test_matching_slices_pass_and_other_tasks_are_skipped():
    ref = load_harness_split("ifeval_k_v1")
    check_ifeval_k_v1("ifeval", 40, 150, list(ref["selection"]), list(ref["test"]))
    check_ifeval_k_v1("gsm8k", 40, 300, [0], [1])


def test_different_slices_raise():
    ref = load_harness_split("ifeval_k_v1")
    sel, test = list(ref["selection"]), list(ref["test"])
    sel[0], test[0] = test[0], sel[0]
    with pytest.raises(RuntimeError, match="ifeval_k_v1"):
        check_ifeval_k_v1("ifeval", 40, 150, sel, test)
