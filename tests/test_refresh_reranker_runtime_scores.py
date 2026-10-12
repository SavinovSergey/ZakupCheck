from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from refresh_reranker_runtime_scores import migrate_scores, missing_pairs  # noqa: E402


def test_migrate_scores_and_find_only_new_pairs() -> None:
    migrated = migrate_scores(
        {"case": {"old-1": 0.5, "removed": 0.1}},
        {"old-1": "new-2"},
    )
    assert migrated == {"case": {"new-2": 0.5}}
    assert missing_pairs(migrated, {"case": {"new-2": 1.0, "new-3": 2.0}}) == {
        "case": ["new-3"]
    }
