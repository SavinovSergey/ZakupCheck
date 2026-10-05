"""Schemas for the hand-labelled FAS evaluation set (DESIGN §4.3, §7)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Topic = Literal[
    "object_description",
    "participant_requirements",
    "notice_content",
    "application_review",
    "contract_terms",
    "contract_conclusion",
    "single_supplier_procedure",
    "competition",
    "national_regime",
]
Outcome = Literal["upheld", "rejected"]
Split = Literal["dev", "eval"]
AnnotationStatus = Literal["verified", "draft", "excluded"]
ExclusionReason = Literal[
    "rejected_argument",
    "relies_on_bylaw",
    "missing_full_decision",
    "missing_gold_norms",
    "procedural_violation_only",
    "ambiguous_norm_mapping",
    "duplicate_case",
    "source_span_unverified",
    "edition_not_applicable",
]


class DocLocator(BaseModel):
    """Где в извещении / приложении лежит doc_fragment (для генерации и faithfulness)."""

    path: str | None = None  # путь относительно data/ или кейса
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    note: str | None = None  # свободная пометка, пока нет оффсетов


class SourceSpan(BaseModel):
    """An exact excerpt from a locally extracted source file."""

    path: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    note: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("source span length must equal len(quote)")
        return self


class GoldNormEvidence(BaseModel):
    """A decision excerpt tying one norm to the argument under review."""

    norm_id: str
    decision_span: SourceSpan


class EvalCase(BaseModel):
    case_id: str
    edition_id: str  # какая редакция корпуса, напр. "2026-08-04"
    topic: Topic
    split: Split
    outcome: Outcome

    # Retrieval query D. Only raw/blind are benchmark inputs; summary is for humans.
    complaint_argument_raw: str | None = None
    complaint_argument_blind: str | None = None
    complaint_argument_summary: str
    complaint_argument_source: SourceSpan | None = None
    gold_norms: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldNormEvidence] = Field(default_factory=list)
    outcome_evidence: SourceSpan | None = None
    relies_on_bylaw: bool = False
    annotation_status: AnnotationStatus = "draft"
    exclusion_reasons: list[ExclusionReason] = Field(default_factory=list)
    case_group_id: str
    duplicate_of: str | None = None

    # --- документ (нужен и retrieval A, и генерации) ---
    doc_fragment: str
    doc_locator: DocLocator | None = None
    doc_paths: list[str] = Field(default_factory=list)

    # --- происхождение ---
    procurement_id: str | None = None
    procurement_url: str | None = None
    notice_date: str | None = None  # YYYY-MM-DD
    fas_decision_id: str | None = None
    fas_decision_url: str | None = None

    # --- генерация / человек (не строгая метрика retrieval) ---
    issue_type: str | None = None  # enum зафиксируем после первых ~10 кейсов
    gold_notes: str | None = None

    @model_validator(mode="after")
    def validate_retrieval_annotation(self) -> "EvalCase":
        if len(set(self.gold_norms)) != len(self.gold_norms):
            raise ValueError("gold_norms must not contain duplicates")
        if len(set(self.exclusion_reasons)) != len(self.exclusion_reasons):
            raise ValueError("exclusion_reasons must not contain duplicates")

        evidence_ids = [item.norm_id for item in self.gold_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("gold_evidence must contain one entry per norm_id")

        if self.annotation_status == "verified":
            required = {
                "complaint_argument_raw": self.complaint_argument_raw,
                "complaint_argument_blind": self.complaint_argument_blind,
                "complaint_argument_source": self.complaint_argument_source,
                "outcome_evidence": self.outcome_evidence,
                "notice_date": self.notice_date,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"verified case is missing: {', '.join(missing)}")
            if self.outcome != "upheld":
                raise ValueError("verified retrieval case must be upheld")
            if self.relies_on_bylaw:
                raise ValueError("verified retrieval case cannot rely on a bylaw")
            if not self.gold_norms:
                raise ValueError("verified retrieval case must have gold_norms")
            if set(evidence_ids) != set(self.gold_norms):
                raise ValueError("verified case needs evidence for every gold_norm")
            if self.exclusion_reasons:
                raise ValueError("verified case cannot have exclusion_reasons")
            if self.duplicate_of:
                raise ValueError("verified case cannot be a duplicate")
        elif not self.exclusion_reasons:
            raise ValueError("draft/excluded case needs at least one exclusion reason")

        if self.duplicate_of and "duplicate_case" not in self.exclusion_reasons:
            raise ValueError("duplicate_of requires duplicate_case exclusion reason")
        return self


class Remark(BaseModel):
    """Выход пайплайна / генерации (DESIGN §2.2). Для eval faithfulness позже."""

    doc_quote: str
    doc_span: DocLocator | None = None
    norm_id: str
    norm_quote: str
    issue_type: str | None = None
    verdict: Literal["violation", "uncertain", "insufficient_evidence"]
    evidence_source: Literal["heuristic", "llm"] | None = None
