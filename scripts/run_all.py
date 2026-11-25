"""Thin wrapper — prefer: python src/run_all.py"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "run_all.py"), run_name="__main__")
