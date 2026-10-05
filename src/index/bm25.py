"""In-memory BM25 over structural law chunks."""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from schemas.corpus import Chunk

_TOKEN_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens of letters and digits. ё is folded to е."""
    folded = text.lower().replace("ё", "е")
    return _TOKEN_RE.findall(folded)


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.model_validate_json(line))
    return chunks


class Bm25Index:
    """BM25Okapi over chunk texts. Built in memory; nothing is written to disk."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 index needs at least one chunk")
        self.chunks = chunks
        self._tokens = [tokenize(chunk.text) or ["_empty"] for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokens)

    @property
    def token_count(self) -> int:
        return sum(len(tokens) for tokens in self._tokens)

    def search(self, query: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [self.chunks[i] for i in order[:k]]
