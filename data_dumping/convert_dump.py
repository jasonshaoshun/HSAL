#!/usr/bin/env python3
"""Compatibility wrapper for the generalized STE dump extractor.

The maintained implementation lives in
``src/data/STE_extract_stackexchange_dump.py``. With no arguments, this wrapper
keeps the old Super User defaults used by ``data_dumping/isambard.sh``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    extractor = repo_root / "src" / "data" / "STE_extract_stackexchange_dump.py"

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--dataset",
                "superuser",
                "--input-dir",
                "superuser",
                "--output",
                "superuser/superuser.jsonl",
            ]
        )

    runpy.run_path(str(extractor), run_name="__main__")


if __name__ == "__main__":
    main()
