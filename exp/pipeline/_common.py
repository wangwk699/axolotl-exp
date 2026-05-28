"""Shared helpers for RQ pipeline runners."""

from __future__ import annotations

import subprocess
import sys


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
