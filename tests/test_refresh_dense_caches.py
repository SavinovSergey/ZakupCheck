from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from schemas.corpus import Chunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from refresh_dense_caches import is_inserted_article_chunk, legacy_view, merge_vectors  # noqa: E402


def _chunk(index: int, norm_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"chk_ed_{index:05d}",
        edition_id="ed",
        chunking="article" if norm_id.endswith("art.6") else "part",
        text=text,
        char_start=index,
        char_end=index + len(text),
        norm_ids=[norm_id],
        max_chunk=700,
        article_title_prefix=False,
    )


def test_refresh_inserts_article_vectors_without_reencoding_legacy_rows() -> None:
    chunks = [
        _chunk(1, "44FZ:ed:art.1:ch.1", "old one"),
        _chunk(2, "44FZ:ed:art.6", "new article"),
        _chunk(3, "44FZ:ed:art.7:ch.1", "old two"),
    ]
    assert is_inserted_article_chunk(chunks[1])
    assert [chunk.chunk_id for chunk in legacy_view(chunks)] == ["chk_ed_00001", "chk_ed_00002"]
    old = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    merged = merge_vectors(chunks, old, {"new article": np.asarray([0.5, 0.5], dtype=np.float32)})
    np.testing.assert_array_equal(
        merged,
        np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32),
    )
