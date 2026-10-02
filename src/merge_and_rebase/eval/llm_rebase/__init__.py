"""LLM rebase entrypoint package (python -m merge_and_rebase.eval.llm_rebase).

Pure move of the former eval/llm_rebase.py module (plus llm_common -> common, lm_harness_runner -> harness,
llm_eval_only -> eval_only). Every name that used to be importable from merge_and_rebase.eval.llm_rebase is
re-exported here.
"""

from . import cli as _cli

globals().update({_n: getattr(_cli, _n) for _n in dir(_cli) if not (_n.startswith("__") and _n.endswith("__"))})
