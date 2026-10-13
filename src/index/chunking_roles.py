"""Fair article-ranking and evidence-packing controls across chunking schemes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from index.recall import article_key
from schemas.corpus import Chunk

Span = tuple[int, int]


def overlap_length(left: Span, right: Span) -> int:
    """Number of source characters shared by two half-open spans."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def overlap_weighted_article_ranking(
    ranked: Sequence[Chunk],
    article_spans: Mapping[str, Span],
    *,
    pool: int = 50,
    rank_k: int = 60,
) -> list[str]:
    """Project chunks to articles without giving boundary slivers a full rank.

    Each article receives its best ``overlap_fraction / (rank_k + rank)`` score.
    Taking the best contribution avoids rewarding a scheme merely for producing
    more chunks from the same article. Ties are deterministic.
    """
    if pool < 0 or rank_k < 0:
        raise ValueError("pool and rank_k must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for rank, chunk in enumerate(ranked[:pool], start=1):
        chunk_span = (chunk.char_start, chunk.char_end)
        chunk_length = max(1, chunk.char_end - chunk.char_start)
        articles = {article_key(norm_id) for norm_id in chunk.norm_ids}
        for article in articles:
            span = article_spans.get(article)
            if span is None:
                continue
            fraction = overlap_length(chunk_span, span) / chunk_length
            if fraction <= 0:
                continue
            score = fraction / (rank_k + rank)
            if score > scores.get(article, -1.0):
                scores[article] = score
                best_rank[article] = rank
    return sorted(scores, key=lambda article: (-scores[article], best_rank[article], article))


def pack_article_evidence(
    ranked: Sequence[Chunk],
    article_ranking: Sequence[str],
    article_spans: Mapping[str, Span],
    *,
    char_budget: int,
) -> list[Chunk]:
    """Pack evidence under a character budget, one representative per article first.

    A multi-article window may represent several selected articles while paying
    its text cost once. Remaining capacity is filled in original retrieval order.
    """
    if char_budget < 0:
        raise ValueError("char_budget must be non-negative")
    selected = list(dict.fromkeys(article_ranking))
    selected_set = set(selected)
    best: dict[str, Chunk] = {}
    for chunk in ranked:
        chunk_span = (chunk.char_start, chunk.char_end)
        for article in selected_set:
            span = article_spans.get(article)
            if span is not None and overlap_length(chunk_span, span) > 0:
                best.setdefault(article, chunk)

    packed: list[Chunk] = []
    seen: set[str] = set()
    used = 0

    def append(chunk: Chunk | None) -> None:
        nonlocal used
        if chunk is None or chunk.chunk_id in seen:
            return
        cost = len(chunk.text)
        if used + cost > char_budget:
            return
        packed.append(chunk)
        seen.add(chunk.chunk_id)
        used += cost

    for article in selected:
        append(best.get(article))
    for chunk in ranked:
        if chunk.chunk_id in seen:
            continue
        chunk_span = (chunk.char_start, chunk.char_end)
        if any(
            span is not None and overlap_length(chunk_span, span) > 0
            for span in (article_spans.get(article) for article in selected_set)
        ):
            append(chunk)
    return packed


def covered_fraction(chunks: Sequence[Chunk], span: Span) -> float:
    """Fraction of a source span covered by the union of retrieved chunks."""
    length = span[1] - span[0]
    if length <= 0:
        return 0.0
    intervals = []
    for chunk in chunks:
        start = max(span[0], chunk.char_start)
        end = min(span[1], chunk.char_end)
        if start < end:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return min(1.0, covered / length)


def outside_selected_fraction(
    chunks: Sequence[Chunk],
    selected_articles: Sequence[str],
    article_spans: Mapping[str, Span],
) -> float:
    """Share of packed characters lying outside the selected article spans."""
    total = sum(max(0, chunk.char_end - chunk.char_start) for chunk in chunks)
    if total == 0:
        return 0.0
    spans = [article_spans[article] for article in set(selected_articles) if article in article_spans]
    inside = 0
    for chunk in chunks:
        chunk_span = (chunk.char_start, chunk.char_end)
        intersections = []
        for span in spans:
            start = max(chunk_span[0], span[0])
            end = min(chunk_span[1], span[1])
            if start < end:
                intersections.append((start, end))
        if not intersections:
            continue
        intersections.sort()
        start, end = intersections[0]
        for next_start, next_end in intersections[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                inside += end - start
                start, end = next_start, next_end
        inside += end - start
    return max(0.0, min(1.0, 1.0 - inside / total))
