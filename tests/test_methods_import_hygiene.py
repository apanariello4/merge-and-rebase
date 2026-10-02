"""Import hygiene for ``rebase/methods`` (Phase 7 S3).

Helpers used by more than one method live in ``methods/_shared.py``. No method module may
import an underscore-prefixed name from a *sibling* method module, ``_shared`` must not import
any method module, and ``theseus`` must keep re-exporting the names it used to expose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import merge_and_rebase.rebase.methods as methods_pkg
from merge_and_rebase.rebase.methods import _shared, theseus

METHODS_DIR = Path(methods_pkg.__file__).parent
ALLOWED_SIBLINGS = {"_shared"}


def _method_files() -> list[Path]:
    files = sorted(p for p in METHODS_DIR.glob("*.py") if p.name != "__init__.py")
    files += sorted((METHODS_DIR / "ariadne").glob("*.py"))
    return files


def _method_modules() -> set[str]:
    mods = {p.stem for p in METHODS_DIR.glob("*.py") if p.name != "__init__.py"}
    mods |= {p.name for p in METHODS_DIR.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    return mods


def _resolve(path: Path, node: ast.ImportFrom) -> str | None:
    """Return the dotted target module of a ``from ... import`` relative to ``merge_and_rebase``."""
    pkg_parts = list(path.relative_to(METHODS_DIR.parents[2]).with_suffix("").parts[:-1])  # module's package
    if node.level == 0:
        return node.module
    base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _violations(path: Path, text: str | None = None) -> list[str]:
    own_pkg = path.parent.name if path.parent != METHODS_DIR else None
    methods_prefix = "merge_and_rebase.rebase.methods"
    siblings = _method_modules()
    out: list[str] = []
    tree = ast.parse(path.read_text() if text is None else text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(methods_prefix + "."):
                    first = alias.name[len(methods_prefix) + 1 :].split(".")[0]
                    if first in siblings and first not in ALLOWED_SIBLINGS and first != own_pkg:
                        out.append(f"{path.name}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(path, node)
            if target is None or not target.startswith(methods_prefix):
                continue
            rest = target[len(methods_prefix) :].lstrip(".")
            parts = rest.split(".") if rest else []
            if not parts:
                # ``from . import theseus`` style: imported names are sibling modules
                for alias in node.names:
                    if alias.name in siblings and alias.name not in ALLOWED_SIBLINGS and alias.name != own_pkg:
                        out.append(f"{path.name}:{node.lineno}: from {target} import {alias.name}")
                continue
            first = parts[0]
            if first in ALLOWED_SIBLINGS or first == own_pkg or first not in siblings:
                continue
            for alias in node.names:
                if alias.name.startswith("_") or alias.name == "*":
                    out.append(f"{path.name}:{node.lineno}: from {target} import {alias.name}")
    return out


def test_no_underscore_imports_from_sibling_methods():
    bad = [v for p in _method_files() for v in _violations(p)]
    assert not bad, "underscore imports from sibling method modules (use methods/_shared.py):\n" + "\n".join(bad)


def test_hygiene_scan_detects_violations():
    # Negative control: the scanner must flag sibling underscore imports and sibling-module aliases.
    path = METHODS_DIR / "bico.py"
    assert _violations(path, "from .theseus import _to_tokens\n")
    assert _violations(path, "from . import theseus as _t\n")
    assert _violations(path, "from merge_and_rebase.rebase.methods.theseus import _to_tokens\n")
    assert not _violations(path, "from ._shared import _to_tokens\n")
    assert not _violations(path, "from .theseus import TheseusRebase\n")
    assert not _violations(path, "from ..base import _anything\n")


def test_shared_does_not_import_method_modules():
    tree = ast.parse(Path(_shared.__file__).read_text())
    siblings = _method_modules() - {"_shared"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module and node.module.split(".")[0] in siblings:
                raise AssertionError(f"_shared imports sibling method {node.module}")
            if node.level == 1 and node.module is None:
                assert not ({a.name for a in node.names} & siblings)
            assert not (node.module or "").startswith("merge_and_rebase.rebase.methods")


def test_theseus_reexports_are_identical_objects():
    for name in (
        "_content_row_mask",
        "_drop_padding_rows",
        "_to_tokens",
        "_interp_2d_tokens",
        "_compute_procrustes_map_from_cov",
        "_transport_weight",
        "_precompute_transforms",
        "_apply_transforms_to_visual_delta",
        "ActivationStore",
        "InterpolatedBlockActivations",
        "_LayerTransform",
    ):
        assert getattr(theseus, name) is getattr(_shared, name), name
