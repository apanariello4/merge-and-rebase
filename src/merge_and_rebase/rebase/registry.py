from __future__ import annotations

from .base import FitMethod, TransportMethod

_METHODS: dict[str, TransportMethod | FitMethod] = {}
# alias -> canonical registered name. An alias resolves to the very same method
# object (e.g. "direct_residual" -> "ariadne"), so every dispatch decision keyed
# on the canonical name is identical for both spellings.
_ALIASES: dict[str, str] = {}


def register(method: TransportMethod | FitMethod) -> None:
    if method.name in _METHODS or method.name in _ALIASES:
        raise KeyError(f"Rebase method '{method.name}' already registered")
    _METHODS[method.name] = method


def register_alias(alias: str, canonical: str) -> None:
    if canonical not in _METHODS:
        raise KeyError(f"Cannot alias '{alias}': unknown rebase method '{canonical}'")
    if alias in _METHODS or alias in _ALIASES:
        raise KeyError(f"Rebase method '{alias}' already registered")
    _ALIASES[alias] = canonical


def canonical_method_name(name: str) -> str:
    """Resolve a registered alias to its canonical name; any other string is returned unchanged."""
    return _ALIASES.get(name, name)


def get_method(name: str) -> TransportMethod | FitMethod:
    canonical = canonical_method_name(name)
    if canonical not in _METHODS:
        raise KeyError(f"Unknown rebase method '{name}'. Available: {list_methods()}")
    return _METHODS[canonical]


def list_methods() -> list[str]:
    """Canonical names and aliases, sorted."""
    return sorted([*_METHODS, *_ALIASES])


# Import built-in method modules for side-effect registration.
from . import methods as _methods  # noqa: F401,E402
