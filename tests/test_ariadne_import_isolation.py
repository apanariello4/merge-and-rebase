"""The Ariadne package is a self-contained rebase method: importing it must not
pull in any run entrypoint from ``merge_and_rebase.eval``."""

from __future__ import annotations

import subprocess
import sys


def test_ariadne_package_imports_nothing_from_eval():
    code = (
        "import sys, merge_and_rebase.rebase.methods.ariadne; "
        "bad=[m for m in sys.modules if m.startswith('merge_and_rebase.eval')]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
