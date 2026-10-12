from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_law_corpus_audit import repeal_basis


def test_repeal_basis_uses_last_law_for_added_then_repealed_article() -> None:
    note = (
        "(Дополнение статьей - Федеральный закон от 31.12.2017 № 504-ФЗ) "
        "(Утратила силу - Федеральный закон от 02.07.2021 № 360-ФЗ)"
    )
    assert repeal_basis(note) == "02.07.2021 № 360-ФЗ"
