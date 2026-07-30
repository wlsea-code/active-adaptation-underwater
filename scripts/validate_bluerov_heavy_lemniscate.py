"""Validate BlueROVHeavy Lemniscate tracking with the shared validator."""

from __future__ import annotations

import sys
import runpy
from pathlib import Path


if not any(argument.startswith("task=") for argument in sys.argv[1:]):
    sys.argv.insert(1, "task=UW/BlueROVHeavyLemniscate")

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("validate_bluerov_lemniscate.py")),
        run_name="__main__",
    )
