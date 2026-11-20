from __future__ import annotations

import pytest

from scripts.run_grounded_dev_comparison import _article_id, _fragment


def test_fragment_anchor_uses_exact_source_offsets(tmp_path):
    source = tmp_path / "notice.txt"
    source.write_text("before\nусловие\nafter", encoding="utf-8")

    spec = _fragment(
        {"path": "notice.txt", "fragment_id": "claim", "role": "claim", "anchor": "условие", "before": 7, "after": 6},
        tmp_path,
    )

    assert source.read_text(encoding="utf-8")[spec.char_start:spec.char_end] == "before\nусловие\nafter"


def test_fragment_rejects_ambiguous_anchor_and_escape(tmp_path):
    root = tmp_path / "notices"
    root.mkdir()
    (root / "notice.txt").write_text("same and same", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        _fragment({"path": "notice.txt", "fragment_id": "claim", "role": "claim", "anchor": "same"}, root)
    with pytest.raises(ValueError, match="escapes documents root"):
        _fragment({"path": "../outside.txt", "fragment_id": "claim", "role": "claim", "anchor": "outside"}, root)


def test_fragment_window_expands_to_complete_lines(tmp_path):
    source = tmp_path / "notice.txt"
    source.write_text("Previous line\nTarget condition in full\nFollowing line\n", encoding="utf-8")

    spec = _fragment(
        {"path": "notice.txt", "fragment_id": "claim", "role": "claim", "anchor": "condition", "before": 0, "after": 0},
        tmp_path,
    )

    assert source.read_text(encoding="utf-8")[spec.char_start:spec.char_end] == "Target condition in full"


def test_article_id_extracts_parent_of_exact_norm():
    assert _article_id("44FZ:2026-08-04:art.31:ch.6") == "44FZ:2026-08-04:art.31"
    with pytest.raises(ValueError, match="invalid norm id"):
        _article_id("44FZ:2026-08-04:ch.6")
