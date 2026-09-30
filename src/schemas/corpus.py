"""Pydantic-схемы корпуса 44-ФЗ (DESIGN §4.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Edition(BaseModel):
    law_id: str = "44FZ"
    edition_id: str
    effective_from: str | None = None
    effective_to: str | None = None
    source_url: str | None = None
    content_sha256: str
    title: str | None = None


class NormUnit(BaseModel):
    norm_id: str
    edition_id: str
    article: str  # "33" или "24.1"
    part: str | None = None
    point: str | None = None
    level: Literal["article", "part", "point"]
    parent_norm_id: str | None = None
    title: str | None = None
    text: str
    char_start: int
    char_end: int
    order: int
    is_retrieval_unit: bool = False


class Chunk(BaseModel):
    chunk_id: str
    edition_id: str
    chunking: Literal["part", "point", "window"]
    text: str
    char_start: int
    char_end: int
    norm_ids: list[str] = Field(min_length=1)
    max_chunk: int | None = None
    article_title_prefix: bool = False
