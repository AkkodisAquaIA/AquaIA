"""Black-box test of the AquaIA CLI entry point.

Invokes `python main.py --help` in a subprocess (rather than calling
`main.main()` directly) so we exercise the exact same path the Dockerfile
CMD takes — argparse construction, subparser wiring, and the import chain
that pulls in `detection.infer` / `detection.train`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_main_help_returns_zero_and_lists_subcommands() -> None:
	result = subprocess.run(
		[sys.executable, "main.py", "--help"],
		cwd=REPO_ROOT,
		capture_output=True,
		text=True,
		timeout=60,
	)
	assert result.returncode == 0, f"`python main.py --help` exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
	# Both subcommands must appear in --help output.
	assert "train" in result.stdout, f"'train' subcommand missing from --help output:\n{result.stdout}"
	assert "infer" in result.stdout, f"'infer' subcommand missing from --help output:\n{result.stdout}"
