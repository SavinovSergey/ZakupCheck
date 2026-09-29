#!/usr/bin/env python3
"""CLI: скачать 44-ФЗ с pravo.gov.ru в data/law/{edition_id}/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.law.download import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
