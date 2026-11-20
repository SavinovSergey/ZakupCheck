"""Input and output contracts for one grounded complaint argument."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.eval_case import Remark


class DocumentFragment(BaseModel):
    fragment_id: str = Field(min_length=1)
    path: str = Field(min_length=1)  # relative to the configured documents root
    role: Literal["claim", "context"] = "claim"
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_span(self) -> "DocumentFragment":
        if self.char_end <= self.char_start:
            raise ValueError("fragment end must exceed start")
        return self


class ArgumentRequest(BaseModel):
    complaint_argument: str = Field(min_length=1)
    edition_id: str = Field(min_length=1)
    fragments: list[DocumentFragment] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fragments(self) -> "ArgumentRequest":
        ids = [fragment.fragment_id for fragment in self.fragments]
        if len(set(ids)) != len(ids):
            raise ValueError("fragment_id values must be unique")
        return self


class AbstainDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["abstain"]
    reason: str = Field(min_length=1, max_length=240)


class RemarkDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["remark"]
    remark_text: str = Field(min_length=1, max_length=350)
    source_fragment_id: str = Field(min_length=1, max_length=64)
    doc_span_id: str = Field(min_length=1, max_length=20)
    norm_chunk_id: str = Field(min_length=1, max_length=100)
    norm_id: str = Field(min_length=1, max_length=100)
    norm_span_id: str = Field(min_length=1, max_length=20)


ModelDraft = Annotated[AbstainDraft | RemarkDraft, Field(discriminator="decision")]


class GroundedResult(BaseModel):
    status: Literal["remark", "insufficient_evidence", "error"]
    remark: Remark | None = None
    reason_code: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def consistent_outcome(self) -> "GroundedResult":
        if self.status == "remark":
            if self.remark is None or self.reason_code is not None:
                raise ValueError("remark status requires a remark and no refusal reason")
        elif self.remark is not None or not self.reason_code:
            raise ValueError("non-remark status requires a reason and no remark")
        return self
