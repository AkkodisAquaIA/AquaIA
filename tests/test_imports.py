"""Import smoke tests.

The goal is purely to guarantee that every top-level package in the repo
loads without raising at import time. This catches:

  * broken refactors that leave a `from foo import bar` dangling
  * missing dependencies in requirements-vm.txt
  * Python version regressions

Heavy modules (model classes, training loops) are not exercised here — they
are covered by `test_smoke_pipeline.py` and by the Docker `--help` smoke in
`test_main.py`.
"""

from __future__ import annotations

import importlib

import pytest

# Top-level packages we ship. `IP/` and `src/` are intentionally excluded:
# they are scratchpad scripts, not packages, and their files do not have
# package semantics (no __init__.py / standalone scripts).
TOP_LEVEL_PACKAGES = [
    "detection",
    "classification",
    "data_cleaning",
    "data_processing",
    "dataloading",
    "dataset_utils",
    "sharepoint_dataloading",
]


@pytest.mark.parametrize("package_name", TOP_LEVEL_PACKAGES)
def test_top_level_package_imports(package_name: str) -> None:
    """Each top-level package imports cleanly."""
    module = importlib.import_module(package_name)
    assert module is not None, f"importlib returned None for {package_name!r}"
