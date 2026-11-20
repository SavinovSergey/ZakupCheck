"""Deterministic short citation spans for source fragments and law chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_SPAN_CHARS = 200
_LEADING_MARKUP = re.compile(r"^(?:\s|__|[-•]|<[^>]+>)*")
_TRAILING_MARKUP = re.compile(r"(?:\s|__|<[^>]+>)*$")


@dataclass(frozen=True)
class QuoteCandidate:
    span_id: str
    start: int
    end: int
    text: str


def quote_candidates(text: str) -> list[QuoteCandidate]:
    """Split at line/word boundaries, retaining exact Unicode offsets."""
    spans: list[QuoteCandidate] = []
    for line in re.finditer(r"[^\r\n]+", text):
        raw_line = line.group()
        visible_start = _LEADING_MARKUP.match(raw_line).end()
        visible_end = _TRAILING_MARKUP.search(raw_line).start()
        if visible_start >= visible_end:
            continue
        words = list(re.finditer(r"\S+", raw_line[visible_start:visible_end]))
        if not words:
            continue
        first = 0
        while first < len(words):
            last = first
            while last + 1 < len(words) and (
                words[last + 1].end() - words[first].start() <= MAX_SPAN_CHARS
            ):
                last += 1
            start = line.start() + visible_start + words[first].start()
            end = line.start() + visible_start + words[last].end()
            spans.append(QuoteCandidate(f"s{len(spans)}", start, end, text[start:end]))
            first = last + 1
    return spans
