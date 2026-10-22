#!/usr/bin/env python3
"""Compatibility wrapper for the unified artifact builder."""
from __future__ import annotations

from artifact_tools.make_artifacts import main


if __name__ == "__main__":
    raise SystemExit(main())
