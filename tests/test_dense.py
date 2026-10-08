"""Dense search on fake vectors. Does not download a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from index.dense import DOCUMENT_PREFIX, QUERY_PREFIX, DenseIndex
from schemas.corpus import Chunk


def _chunk(chunk_id: str, text: str, norm_ids: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="2026-08-04",
        chunking="point",
        text=text,
        char_start=0,
        char_end=len(text),
        norm_ids=norm_ids,
        max_chunk=1500,
        article_title_prefix=False,
    )


class RecordingEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, list[str]]] = []
        self.doc_calls = 0

    def encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        self.calls.append((prefix, list(texts)))
        if prefix == DOCUMENT_PREFIX:
            self.doc_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def truncated_count(self, texts: list[str]) -> int:
        return 0


def test_search_returns_nearest_chunk_and_distinct_prefixes() -> None:
    encoder = RecordingEncoder(
        {
            "товарный знак": [1.0, 0.0],
            "участники": [0.0, 1.0],
            "контракт": [-1.0, 0.0],
            "знак": [0.9, 0.1],
        }
    )
    index = DenseIndex(
        [
            _chunk("a", "товарный знак", ["44FZ:2026-08-04:art.33:ch.1:p.1"]),
            _chunk("b", "участники", ["44FZ:2026-08-04:art.31:ch.1"]),
            _chunk("c", "контракт", ["44FZ:2026-08-04:art.34:ch.1"]),
        ],
        encoder,
    )
    hits = index.search("знак", k=1)
    assert hits[0].norm_ids == ["44FZ:2026-08-04:art.33:ch.1:p.1"]
    assert encoder.calls[0][0] == DOCUMENT_PREFIX
    assert encoder.calls[1] == (QUERY_PREFIX, ["знак"])


def test_search_uses_prefixes_stored_on_the_encoder() -> None:
    encoder = RecordingEncoder({"товарный знак": [1.0, 0.0], "знак": [0.9, 0.1]})
    encoder.query_prefix = "Represent this sentence for searching relevant passages: "
    encoder.document_prefix = ""
    index = DenseIndex(
        [_chunk("a", "товарный знак", ["44FZ:2026-08-04:art.33:ch.1:p.1"])],
        encoder,
    )
    index.search("знак", k=1)
    assert encoder.calls[0] == ("", ["товарный знак"])
    assert encoder.calls[1][0] == "Represent this sentence for searching relevant passages: "


def test_equal_scores_keep_earlier_chunk() -> None:
    encoder = RecordingEncoder({"один": [1.0, 0.0], "два": [1.0, 0.0], "запрос": [1.0, 0.0]})
    index = DenseIndex(
        [
            _chunk("a", "один", ["44FZ:2026-08-04:art.33"]),
            _chunk("b", "два", ["44FZ:2026-08-04:art.42"]),
        ],
        encoder,
    )
    hits = index.search("запрос", k=1)
    assert hits[0].chunk_id == "a"


def test_cache_skips_second_document_encode(tmp_path: Path) -> None:
    chunks = [
        _chunk("a", "товарный знак", ["44FZ:2026-08-04:art.33"]),
        _chunk("b", "участники", ["44FZ:2026-08-04:art.31"]),
    ]
    encoder = RecordingEncoder({"товарный знак": [1.0, 0.0], "участники": [0.0, 1.0], "знак": [1.0, 0.0]})
    first = DenseIndex.build(
        chunks,
        encoder,
        cache_dir=tmp_path,
        model_name="fake/model",
        max_chunk=300,
    )
    second = DenseIndex.build(
        chunks,
        encoder,
        cache_dir=tmp_path,
        model_name="fake/model",
        max_chunk=300,
    )
    assert encoder.doc_calls == 1
    assert first.search("знак", k=1)[0].chunk_id == second.search("знак", k=1)[0].chunk_id
