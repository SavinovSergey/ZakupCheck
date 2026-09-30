"""Схема золотого кейса eval (DESIGN §4.3, §7).

Retrieval сейчас использует: complaint_argument, gold_norms, outcome, relies_on_bylaw, topic, split.
Поля под генерацию / faithfulness заложены заранее (issue_type, doc_locator, …).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Topic = Literal["ooz_33", "participants_31", "notice_42", "contract_34"]
Outcome = Literal["upheld", "rejected"]
Split = Literal["dev", "eval"]


class DocLocator(BaseModel):
    """Где в извещении / приложении лежит doc_fragment (для генерации и faithfulness)."""

    path: str | None = None  # путь относительно data/ или кейса
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    note: str | None = None  # свободная пометка, пока нет оффсетов


class EvalCase(BaseModel):
    case_id: str
    edition_id: str  # какая редакция корпуса, напр. "2026-08-04"
    topic: Topic
    split: Split
    outcome: Outcome

    # --- retrieval (режим D) ---
    complaint_argument: str
    gold_norms: list[str] = Field(default_factory=list)
    # norm_id в каноне §4.2; для rejected обычно [].
    # Строгий retrieval-eval: outcome=upheld и relies_on_bylaw=False.

    relies_on_bylaw: bool = False

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


class Remark(BaseModel):
    """Выход пайплайна / генерации (DESIGN §2.2). Для eval faithfulness позже."""

    doc_quote: str
    doc_span: DocLocator | None = None
    norm_id: str
    norm_quote: str
    issue_type: str | None = None
    verdict: Literal["violation", "uncertain", "insufficient_evidence"]
    evidence_source: Literal["heuristic", "llm"] | None = None
