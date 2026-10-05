"""Analysis utilities that sit downstream of a run's saved artifacts.

Unlike ``eval/`` (which produces results) or ``merge/``/``rebase/`` (which
produce merged or transported checkpoints), this package holds read-only
diagnostics computed from already-saved outputs -- e.g. representation
similarity between two models' activations. Nothing here fits a transport map
or searches a hyperparameter.
"""
