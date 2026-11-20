"""One-argument pipeline: retrieve, draft, check both citations, or abstain."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from generate.llama_server import GenerationError
from index.grounded_retrieval import RetrievalEvidence
from schemas.corpus import Chunk, NormUnit
from schemas.grounded import ArgumentRequest, GroundedResult, ModelDraft
from validate.grounding import LoadedFragment, load_fragments, validate_draft


class Retriever(Protocol):
    def retrieve(self, query: str) -> RetrievalEvidence: ...


class Generator(Protocol):
    def generate(
        self,
        complaint_argument: str,
        fragments: list[LoadedFragment],
        evidence: list[Chunk],
    ) -> ModelDraft: ...


def load_norm_units(path: Path) -> dict[str, NormUnit]:
    units = (
        NormUnit.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {unit.norm_id: unit for unit in units}


class GroundedArgumentPipeline:
    def __init__(
        self,
        *,
        edition_dir: Path,
        documents_root: Path,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self.edition_dir = edition_dir
        self.documents_root = documents_root
        self.retriever = retriever
        self.generator = generator
        self.norm_units = load_norm_units(edition_dir / "norm_units.jsonl")

    def run(self, item: ArgumentRequest) -> tuple[GroundedResult, dict]:
        trace: dict = {
            "edition_id": item.edition_id,
            "complaint_argument": item.complaint_argument,
            "fragments": [],
            "retrieval": None,
            "draft": None,
            "timings_ms": {},
        }
        try:
            if not self.norm_units or any(
                unit.edition_id != item.edition_id for unit in self.norm_units.values()
            ):
                raise ValueError("requested edition does not match the loaded corpus")
            fragments = load_fragments(item, self.documents_root)
            trace["fragments"] = [
                {**fragment.spec.model_dump(), "text": fragment.text}
                for fragment in fragments
            ]
        except (OSError, ValueError) as exc:
            result = GroundedResult(status="error", reason_code="invalid_input", detail=str(exc))
            trace["result"] = result.model_dump(mode="json")
            return result, trace

        if not any(fragment.spec.role == "claim" for fragment in fragments):
            result = GroundedResult(
                status="insufficient_evidence", reason_code="no_claim_fragment"
            )
            trace["result"] = result.model_dump(mode="json")
            return result, trace

        started = time.perf_counter()
        try:
            retrieved = self.retriever.retrieve(item.complaint_argument)
            trace["timings_ms"]["retrieval"] = round((time.perf_counter() - started) * 1000, 1)
            trace["retrieval"] = {
                "article_ids": retrieved.article_ids,
                "scored_chunk_count": len(retrieved.scored_chunk_ids),
                "evidence": [chunk.model_dump(mode="json") for chunk in retrieved.chunks],
            }
            if not retrieved.chunks:
                result = GroundedResult(
                    status="insufficient_evidence", reason_code="no_retrieval_evidence"
                )
            else:
                started = time.perf_counter()
                draft = self.generator.generate(
                    item.complaint_argument, fragments, retrieved.chunks
                )
                trace["timings_ms"]["generation"] = round(
                    (time.perf_counter() - started) * 1000, 1
                )
                trace["draft"] = draft.model_dump(mode="json")
                generation_metadata = getattr(self.generator, "last_generation_metadata", None)
                if generation_metadata is not None:
                    trace["generation_metadata"] = generation_metadata
                result = validate_draft(
                    draft, item, fragments, self.norm_units, retrieved.chunks
                )
        except GenerationError as exc:
            result = GroundedResult(status="error", reason_code="generation_error", detail=str(exc))
        except Exception as exc:
            result = GroundedResult(status="error", reason_code="pipeline_error", detail=str(exc))
        trace["result"] = result.model_dump(mode="json")
        return result, trace
