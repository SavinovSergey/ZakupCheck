"""Deterministic citation checks; these do not assert legal entailment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from schemas.corpus import Chunk, NormUnit
from schemas.eval_case import DocLocator, Remark
from schemas.grounded import ArgumentRequest, DocumentFragment, GroundedResult, ModelDraft
from validate.quote_candidates import quote_candidates

_DOUBLE_QUOTES = str.maketrans({char: '"' for char in '«»“”„‟'})
_SINGLE_QUOTES = str.maketrans({char: "'" for char in '‘’ʼ`'})
_HYPHENS = str.maketrans({char: '-' for char in '‐‑‒–—−'})
MIN_QUOTE_CHARS = 20


@dataclass(frozen=True)
class LoadedFragment:
    spec: DocumentFragment
    text: str
    source_path: Path


def load_fragments(request: ArgumentRequest, documents_root: Path) -> list[LoadedFragment]:
    """Read exact source slices; never accept a path outside the notice root."""
    root = documents_root.resolve(strict=True)
    loaded: list[LoadedFragment] = []
    for spec in request.fragments:
        relative = Path(spec.path)
        if relative.is_absolute():
            raise ValueError(f"absolute document path is not allowed: {spec.path}")
        source_path = (root / relative).resolve(strict=True)
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError(f"document path escapes the source root: {spec.path}")
        source_text = source_path.read_text(encoding="utf-8")
        if spec.char_end > len(source_text):
            raise ValueError(f"fragment exceeds source length: {spec.fragment_id}")
        loaded.append(
            LoadedFragment(spec, source_text[spec.char_start:spec.char_end], source_path)
        )
    return loaded


def _normalized_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Fold only typography/whitespace and preserve original character offsets."""
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] == " ":
                ends[-1] = index + 1
                continue
            normalized = " "
        else:
            normalized = char.translate(_DOUBLE_QUOTES).translate(_SINGLE_QUOTES).translate(_HYPHENS)
        chars.append(normalized)
        starts.append(index)
        ends.append(index + 1)
    return "".join(chars), starts, ends


def quote_span(quote: str, source: str) -> tuple[int, int] | None:
    """Find a contiguous quote after the DESIGN-approved minimal normalization."""
    needle = _normalized_with_offsets(quote)[0].strip()
    if not needle:
        return None
    haystack, starts, ends = _normalized_with_offsets(source)
    index = haystack.find(needle)
    if index < 0:
        return None
    return starts[index], ends[index + len(needle) - 1]


def validate_draft(
    draft: ModelDraft,
    request: ArgumentRequest,
    fragments: list[LoadedFragment],
    norm_units: dict[str, NormUnit],
    evidence: list[Chunk],
) -> GroundedResult:
    """Either emit a fully citation-grounded candidate or a typed refusal."""
    if draft.decision == "abstain":
        return GroundedResult(
            status="insufficient_evidence", reason_code="model_abstained", detail=draft.reason
        )

    def refuse(reason: str) -> GroundedResult:
        return GroundedResult(status="insufficient_evidence", reason_code=reason)

    if not all((draft.remark_text.strip(), draft.source_fragment_id.strip(), draft.norm_id.strip())):
        return refuse("incomplete_remark")
    by_id = {fragment.spec.fragment_id: fragment for fragment in fragments}
    fragment = by_id.get(draft.source_fragment_id)
    if fragment is None:
        return refuse("unknown_source_fragment")
    if fragment.spec not in request.fragments:
        return refuse("unknown_source_fragment")
    if fragment.spec.role != "claim":
        return refuse("doc_quote_not_claim_evidence")
    doc_span = next(
        (span for span in quote_candidates(fragment.text) if span.span_id == draft.doc_span_id),
        None,
    )
    if doc_span is None:
        return refuse("doc_span_not_in_fragment")
    source_text = fragment.source_path.read_text(encoding="utf-8")
    start = fragment.spec.char_start + doc_span.start
    end = fragment.spec.char_start + doc_span.end
    if source_text[fragment.spec.char_start:fragment.spec.char_end] != fragment.text:
        return refuse("source_changed")
    if source_text[start:end] != doc_span.text:
        return refuse("doc_quote_not_in_source")

    norm = norm_units.get(draft.norm_id)
    if norm is None or norm.edition_id != request.edition_id or norm.is_repealed:
        return refuse("unknown_or_wrong_edition_norm")
    chunk = next((item for item in evidence if item.chunk_id == draft.norm_chunk_id), None)
    if chunk is None or draft.norm_id not in chunk.norm_ids:
        return refuse("norm_not_exposed")
    norm_span = next(
        (span for span in quote_candidates(chunk.text) if span.span_id == draft.norm_span_id),
        None,
    )
    if norm_span is None:
        return refuse("norm_span_not_in_evidence")
    if min(len(doc_span.text.strip()), len(norm_span.text.strip())) < MIN_QUOTE_CHARS:
        return refuse("quote_too_short")
    if quote_span(norm_span.text, norm.text) is None:
        return refuse("norm_quote_not_in_unit")
    if chunk.text[norm_span.start:norm_span.end] != norm_span.text:
        return refuse("norm_quote_not_in_evidence")

    return GroundedResult(
        status="remark",
        remark=Remark(
            remark_text=draft.remark_text.strip(),
            doc_quote=doc_span.text,
            doc_span=DocLocator(path=str(fragment.source_path), char_start=start, char_end=end),
            norm_id=draft.norm_id,
            norm_quote=norm_span.text,
            issue_type=None,
            verdict="violation",
            evidence_source="llm",
        ),
    )
