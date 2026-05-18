"""Shared pytest fixtures.

Kept intentionally minimal: pytest's `rootdir` discovery (driven by
`[tool.pytest.ini_options]` in pyproject.toml) already puts the repository
root on sys.path, so `import detection`, `import classification`, etc. work
out of the box. We only add the explicit insert as a defensive fallback for
contributors who invoke pytest from non-standard working directories.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))
