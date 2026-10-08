"""In-memory dense retrieval over structural law chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from schemas.corpus import Chunk

QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "
MAX_TOKENS = 512


class TextEncoder(Protocol):
    def encode(self, texts: list[str], *, prefix: str) -> np.ndarray: ...

    def truncated_count(self, texts: list[str]) -> int: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(b"\0")
        digest.update(chunk.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_paths(cache_dir: Path, model_name: str, max_chunk: int) -> tuple[Path, Path]:
    slug = model_name.replace("/", "__")
    stem = f"{slug}_m{max_chunk}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.meta.json"


def encoder_prefix(encoder: TextEncoder, name: str, default: str) -> str:
    """Prefix stored on the encoder, or the module default when the encoder has none."""
    value = getattr(encoder, name, None)
    return default if value is None else value


class SentenceTransformerEncoder:
    """Local sentence-transformers model. Documents and queries use different prefixes."""

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = QUERY_PREFIX,
        document_prefix: str = DOCUMENT_PREFIX,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_tokens = max_tokens
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = max_tokens

    def encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        if not texts:
            raise ValueError("encode() needs at least one text")
        # Empty string suppresses the model's own default prompt.
        # None would let the library fill that prompt back in.
        vectors = self.model.encode(
            texts,
            prompt=prefix if prefix else "",
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def truncated_count(self, texts: list[str]) -> int:
        """How many document texts exceed max_tokens once the document prefix is attached."""
        import warnings

        tokenizer = self.model.tokenizer
        prefixed = [self.document_prefix + text for text in texts]
        count = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length")
            for start in range(0, len(prefixed), 256):
                encoded = tokenizer(
                    prefixed[start : start + 256],
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )
                count += sum(len(ids) > self.max_tokens for ids in encoded["input_ids"])
        return count


class DenseIndex:
    """Cosine search over L2-normalized chunk vectors. Ties break toward the earlier chunk."""

    def __init__(
        self,
        chunks: list[Chunk],
        encoder: TextEncoder,
        *,
        vectors: np.ndarray | None = None,
        truncated: int = 0,
    ) -> None:
        if not chunks:
            raise ValueError("dense index needs at least one chunk")
        self.chunks = chunks
        self.truncated = truncated
        self._encoder = encoder
        if vectors is None:
            document_prefix = encoder_prefix(encoder, "document_prefix", DOCUMENT_PREFIX)
            vectors = encoder.encode([chunk.text for chunk in chunks], prefix=document_prefix)
        matrix = normalize(vectors)
        if matrix.shape[0] != len(chunks):
            raise ValueError(f"expected {len(chunks)} vectors, got {matrix.shape[0]}")
        self._vectors = matrix

    @classmethod
    def load_cached(
        cls,
        chunks: list[Chunk],
        encoder: TextEncoder,
        *,
        cache_dir: Path,
        model_name: str,
        max_chunk: int,
    ) -> DenseIndex | None:
        """Return an index from the embedding cache. Does not encode documents."""
        fingerprint = corpus_fingerprint(chunks)
        npy_path, meta_path = cache_paths(cache_dir, model_name, max_chunk)
        cached = _load_cache(npy_path, meta_path, model_name, fingerprint)
        if cached is None:
            return None
        vectors, truncated = cached
        return cls(chunks, encoder, vectors=vectors, truncated=truncated)

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        encoder: TextEncoder,
        *,
        cache_dir: Path,
        model_name: str,
        max_chunk: int,
    ) -> DenseIndex:
        cached = cls.load_cached(
            chunks,
            encoder,
            cache_dir=cache_dir,
            model_name=model_name,
            max_chunk=max_chunk,
        )
        if cached is not None:
            return cached
        fingerprint = corpus_fingerprint(chunks)
        npy_path, meta_path = cache_paths(cache_dir, model_name, max_chunk)
        document_prefix = encoder_prefix(encoder, "document_prefix", DOCUMENT_PREFIX)
        truncated = encoder.truncated_count([chunk.text for chunk in chunks])
        vectors = encoder.encode([chunk.text for chunk in chunks], prefix=document_prefix)
        _save_cache(npy_path, meta_path, model_name, fingerprint, vectors, truncated)
        return cls(chunks, encoder, vectors=vectors, truncated=truncated)

    def search(self, query: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        query_prefix = encoder_prefix(self._encoder, "query_prefix", QUERY_PREFIX)
        query_vec = normalize(self._encoder.encode([query], prefix=query_prefix))[0]
        scores = self._vectors @ query_vec
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [self.chunks[i] for i in order[:k]]


def _load_cache(
    npy_path: Path,
    meta_path: Path,
    model_name: str,
    fingerprint: str,
) -> tuple[np.ndarray, int] | None:
    if not npy_path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != model_name or meta.get("fingerprint") != fingerprint:
        return None
    return np.load(npy_path), int(meta["truncated"])


def _save_cache(
    npy_path: Path,
    meta_path: Path,
    model_name: str,
    fingerprint: str,
    vectors: np.ndarray,
    truncated: int,
) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, np.asarray(vectors, dtype=np.float32))
    meta_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "fingerprint": fingerprint,
                "truncated": truncated,
                "rows": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
