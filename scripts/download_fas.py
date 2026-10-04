#!/usr/bin/env python3
"""CLI: жалоба или извещение ЕИС по номеру → data/raw/ + сырой текст."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fas.download import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
