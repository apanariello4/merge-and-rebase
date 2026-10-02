"""LLM rebase entrypoint package (python -m merge_and_rebase.eval.llm_rebase).

Pure move of the former eval/llm_rebase.py module (plus llm_common -> common, lm_harness_runner -> harness,
llm_eval_only -> eval_only). Every name that used to be importable from merge_and_rebase.eval.llm_rebase is
re-exported here.
"""

from collections.abc import Iterable, Mapping  # noqa: F401

from ...data.llm_calibration import resolve_calibration_texts, tokenization_stats  # noqa: F401
from ...hyperparam_search import SearchEvaluation, describe_candidate, summarize_search_results  # noqa: F401
from ...merge.runtime import apply_delta  # noqa: F401
from . import cli as _cli
from .common import default_prompt_for_task, inject_task_head, normalized_acc  # noqa: F401
from .merge import _delta_norm  # noqa: F401

globals().update({_n: getattr(_cli, _n) for _n in dir(_cli) if not (_n.startswith("__") and _n.endswith("__"))})
