#!/usr/bin/env python3
"""Refresh dense caches after adding article-level chunks.

The corpus coverage fix only inserts chunks for active articles whose body has
no numbered parts.  This script proves that removing those chunks and restoring
the old sequential chunk ids reproduces the fingerprint stored in every stale
cache.  Only then does it reuse the old vectors and encode the inserted texts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import load_chunks  # noqa: E402
from index.dense import (  # noqa: E402
    SentenceTransformerEncoder,
    corpus_fingerprint,
    normalize,
)
from index.eval_tables import chunks_path  # noqa: E402
from index.recall import article_key  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from schemas.corpus import Chunk, NormUnit  # noqa: E402

ARTICLE_NORM_RE = re.compile(r"^44FZ:[^:]+:art\.\d+(?:\.\d+)?$")
SIZE_RE = re.compile(r"_m(\d+)\.meta\.json$")


def is_inserted_article_chunk(chunk: Chunk) -> bool:
    return len(chunk.norm_ids) == 1 and bool(ARTICLE_NORM_RE.fullmatch(chunk.norm_ids[0]))


def legacy_view(chunks: list[Chunk]) -> list[Chunk]:
    """Reconstruct the exact chunk sequence represented by the stale cache."""
    retained = [chunk for chunk in chunks if not is_inserted_article_chunk(chunk)]
    return [
        chunk.model_copy(update={"chunk_id": f"chk_{chunk.edition_id}_{index:05d}"})
        for index, chunk in enumerate(retained, start=1)
    ]


def title_prefixed(chunks: list[Chunk], edition_dir: Path) -> list[Chunk]:
    titles: dict[str, str] = {}
    for line in (edition_dir / "norm_units.jsonl").read_text(encoding="utf-8").splitlines():
        unit = NormUnit.model_validate_json(line)
        if unit.level == "article" and unit.title:
            titles[article_key(unit.norm_id)] = unit.title
    result: list[Chunk] = []
    for chunk in chunks:
        articles = sorted({article_key(norm_id) for norm_id in chunk.norm_ids})
        title = next((titles[item] for item in articles if item in titles), None)
        text = f"{title}.\n{chunk.text}" if title else chunk.text
        result.append(chunk.model_copy(update={"text": text, "article_title_prefix": bool(title)}))
    return result


def merge_vectors(
    chunks: list[Chunk],
    old_vectors: np.ndarray,
    inserted_by_text: dict[str, np.ndarray],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    old_index = 0
    for chunk in chunks:
        if is_inserted_article_chunk(chunk):
            rows.append(inserted_by_text[chunk.text])
        else:
            rows.append(old_vectors[old_index])
            old_index += 1
    if old_index != len(old_vectors):
        raise AssertionError(f"used {old_index} old rows out of {len(old_vectors)}")
    return np.asarray(rows, dtype=np.float32)


def compatibility_cosines(
    encoder: SentenceTransformerEncoder,
    legacy: list[Chunk],
    old_vectors: np.ndarray,
) -> list[float]:
    indexes = sorted({0, len(legacy) // 2, len(legacy) - 1})
    texts = [legacy[index].text for index in indexes]
    current = normalize(encoder.encode(texts, prefix=encoder.document_prefix))
    cached = normalize(old_vectors[indexes])
    return [float(np.dot(left, right)) for left, right in zip(current, cached, strict=True)]


def atomic_save(path: Path, vectors: np.ndarray, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".meta.json")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temp_npy = Path(handle.name)
        np.save(handle, vectors)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        temp_meta = Path(handle.name)
        json.dump(meta, handle, ensure_ascii=False)
    os.replace(temp_npy, path)
    os.replace(temp_meta, meta_path)


def refresh_cache(
    *,
    chunks: list[Chunk],
    npy_path: Path,
    meta_path: Path,
    encoder: SentenceTransformerEncoder,
    encoded_texts: dict[str, np.ndarray],
    min_cosine: float,
) -> dict[str, object]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    legacy = legacy_view(chunks)
    if corpus_fingerprint(legacy) != meta.get("fingerprint"):
        raise AssertionError(f"legacy fingerprint mismatch: {meta_path}")
    old_vectors = np.load(npy_path)
    if old_vectors.shape[0] != len(legacy):
        raise AssertionError(
            f"legacy row mismatch for {npy_path}: {old_vectors.shape[0]} != {len(legacy)}"
        )

    cosines = compatibility_cosines(encoder, legacy, old_vectors)
    if min(cosines) < min_cosine:
        raise AssertionError(
            f"model compatibility failed for {npy_path}: min cosine={min(cosines):.8f}"
        )

    inserted = [chunk for chunk in chunks if is_inserted_article_chunk(chunk)]
    missing_texts = list(dict.fromkeys(chunk.text for chunk in inserted if chunk.text not in encoded_texts))
    if missing_texts:
        encoded = encoder.encode(missing_texts, prefix=encoder.document_prefix)
        encoded_texts.update(zip(missing_texts, encoded, strict=True))
    vectors = merge_vectors(chunks, old_vectors, encoded_texts)
    if vectors.shape[1] != old_vectors.shape[1]:
        raise AssertionError(f"embedding dimension changed for {npy_path}")

    new_meta = {
        "model": meta["model"],
        "fingerprint": corpus_fingerprint(chunks),
        "truncated": encoder.truncated_count([chunk.text for chunk in chunks]),
        "rows": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "refresh": {
            "strategy": "reuse-verified-unchanged-vectors-and-encode-inserted-article-chunks",
            "reused_rows": int(old_vectors.shape[0]),
            "encoded_rows": len(inserted),
            "compatibility_min_cosine": min(cosines),
        },
    }
    atomic_save(npy_path, vectors, new_meta)
    return new_meta


def cache_specs(edition_dir: Path, include_title_prefix: bool) -> dict[str, list[tuple[int, Path, Path, bool]]]:
    result: dict[str, list[tuple[int, Path, Path, bool]]] = {}
    directories = [(edition_dir / "dense", False)]
    if include_title_prefix:
        directories.append((edition_dir / "dense_title_prefix", True))
    for directory, prefixed in directories:
        for meta_path in sorted(directory.glob("*.meta.json")):
            match = SIZE_RE.search(meta_path.name)
            if not match:
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model = str(meta["model"])
            npy_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".npy")
            result.setdefault(model, []).append((int(match.group(1)), npy_path, meta_path, prefixed))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh dense caches after article coverage fix")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--skip-title-prefix", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args(argv)

    specs = cache_specs(args.edition_dir, not args.skip_title_prefix)
    selected = args.models or sorted(specs)
    for model in selected:
        if model not in specs:
            raise ValueError(f"no stale caches found for {model}")
        profile = profile_for(model)
        print(f"loading {model}", file=sys.stderr)
        encoder = SentenceTransformerEncoder(
            model,
            query_prefix=profile.query_prefix,
            document_prefix=profile.document_prefix,
            max_tokens=profile.max_tokens,
        )
        encoded_texts: dict[str, np.ndarray] = {}
        for size, npy_path, meta_path, prefixed in sorted(specs[model]):
            chunks = load_chunks(chunks_path(args.edition_dir, size))
            if prefixed:
                chunks = title_prefixed(chunks, args.edition_dir)
            refreshed = refresh_cache(
                chunks=chunks,
                npy_path=npy_path,
                meta_path=meta_path,
                encoder=encoder,
                encoded_texts=encoded_texts,
                min_cosine=args.min_cosine,
            )
            label = "title-prefix" if prefixed else "base"
            info = refreshed["refresh"]
            print(
                f"{model} m{size} {label}: rows={refreshed['rows']} "
                f"reused={info['reused_rows']} encoded={info['encoded_rows']} "
                f"cos={info['compatibility_min_cosine']:.8f}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
