#!/usr/bin/env python3
"""Unified paper artifact builder.

Examples:
  python make_artifacts.py tables
  python artifact_tools/make_artifacts.py tables
  python make_artifacts.py figures
  python make_artifacts.py all
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact_tools.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
