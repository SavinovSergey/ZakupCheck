"""Offline tests of citation gates and one-argument orchestration."""

from __future__ import annotations

import json
from urllib.error import URLError

import pytest
from pydantic import TypeAdapter, ValidationError

from generate.llama_server import GenerationError, LlamaServerGenerator, build_messages, draft_schema
from index.grounded_retrieval import FrozenArticleRetriever, RetrievalEvidence
from pipeline import GroundedArgumentPipeline
from schemas.corpus import Chunk, NormUnit
from schemas.grounded import AbstainDraft, ArgumentRequest, DocumentFragment, ModelDraft, RemarkDraft
from validate.grounding import load_fragments, quote_span, validate_draft
from validate.quote_candidates import quote_candidates

NORM_ID = "44FZ:2026-08-04:art.31:ch.6"
NORM_TEXT = "6. Заказчики не вправе устанавливать требования к участникам закупок в нарушение требований настоящего Федерального закона."
DOC_TEXT = "Предмет: устройство СКУД.\nТребуется лицензия на монтаж пожарной сигнализации."


def sample(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notice.txt").write_text(DOC_TEXT, encoding="utf-8")
    request = ArgumentRequest(
        complaint_argument="Лицензия на пожарную сигнализацию для СКУД избыточна",
        edition_id="2026-08-04",
        fragments=[DocumentFragment(
            fragment_id="requirement", path="notice.txt", char_start=0, char_end=len(DOC_TEXT)
        )],
    )
    norm = NormUnit(
        norm_id=NORM_ID, edition_id="2026-08-04", article="31", part="6",
        level="part", text=NORM_TEXT, char_start=0, char_end=len(NORM_TEXT), order=1,
    )
    chunk = Chunk(
        chunk_id="c31", edition_id="2026-08-04", chunking="part", text=NORM_TEXT,
        char_start=0, char_end=len(NORM_TEXT), norm_ids=[NORM_ID], max_chunk=700,
    )
    draft = RemarkDraft(
        decision="remark", remark_text="Проверьте требование лицензии для работ по СКУД.",
        source_fragment_id="requirement", doc_span_id="s1",
        norm_chunk_id="c31", norm_id=NORM_ID, norm_span_id="s0",
    )
    return request, load_fragments(request, documents), {NORM_ID: norm}, [chunk], draft, documents


def test_quote_normalization_is_typographic_not_paraphrase():
    assert quote_span('«Текст» — один\nабзац', 'Здесь "Текст" - один   абзац.') == (6, 28)
    assert quote_span("иной абзац", "здесь один абзац") is None


def test_valid_quotation_returns_located_candidate(tmp_path):
    item, fragments, norms, evidence, draft, _ = sample(tmp_path)
    result = validate_draft(draft, item, fragments, norms, evidence)
    assert result.status == "remark"
    assert result.remark.verdict == "violation"
    assert result.remark.issue_type is None
    assert result.remark.doc_span.char_start == DOC_TEXT.index("Требуется")
    assert result.remark.doc_span.char_end == len(DOC_TEXT)
    assert result.remark.doc_quote == "Требуется лицензия на монтаж пожарной сигнализации."
    assert result.remark.norm_quote == NORM_TEXT


def test_draft_variants_are_mutually_exclusive():
    adapter = TypeAdapter(ModelDraft)
    assert adapter.validate_python({"decision": "abstain", "reason": "Мало фактов"}) == AbstainDraft(
        decision="abstain", reason="Мало фактов"
    )
    with pytest.raises(ValidationError):
        adapter.validate_python({"decision": "abstain", "reason": "Мало фактов", "remark_text": "Нарушение"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"decision": "remark", "reason": "Мало фактов"})
    assert len(adapter.json_schema()["oneOf"]) == 2


def test_quote_candidates_are_exact_unicode_slices():
    text = "  Первая строка.\n  __- __<a id=\"x\"></a>__Длинный фрагмент «с цитатой» — и ещё словами.__"
    candidates = quote_candidates(text)
    assert [span.text for span in candidates] == [
        "Первая строка.", "Длинный фрагмент «с цитатой» — и ещё словами."
    ]
    assert all(text[span.start:span.end] == span.text for span in candidates)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"doc_span_id": "s100"}, "doc_span_not_in_fragment"),
        ({"norm_id": "44FZ:2026-08-04:art.99"}, "unknown_or_wrong_edition_norm"),
        ({"norm_id": "44FZ:2025-01-01:art.31:ch.6"}, "unknown_or_wrong_edition_norm"),
        ({"norm_span_id": "s100"}, "norm_span_not_in_evidence"),
        ({"norm_chunk_id": "not-shown"}, "norm_not_exposed"),
        ({"source_fragment_id": "unrelated"}, "unknown_source_fragment"),
        ({"remark_text": ""}, "incomplete_remark"),
    ],
)
def test_invalid_or_abstained_drafts_are_refusals(tmp_path, updates, expected):
    item, fragments, norms, evidence, draft, _ = sample(tmp_path)
    result = validate_draft(draft.model_copy(update=updates), item, fragments, norms, evidence)
    assert result.status == "insufficient_evidence"
    assert result.reason_code == expected
    assert result.remark is None


def test_model_abstention_is_refusal(tmp_path):
    item, fragments, norms, evidence, _, _ = sample(tmp_path)
    result = validate_draft(
        AbstainDraft(decision="abstain", reason="Не хватает сведений"),
        item, fragments, norms, evidence,
    )
    assert result.reason_code == "model_abstained"


def test_norm_must_have_been_shown_to_model(tmp_path):
    item, fragments, norms, evidence, draft, _ = sample(tmp_path)
    wrong = evidence[0].model_copy(update={"norm_ids": ["44FZ:2026-08-04:art.32"]})
    assert validate_draft(draft, item, fragments, norms, [wrong]).reason_code == "norm_not_exposed"
    no_quote = evidence[0].model_copy(update={"text": "Совсем другой текст нормы, который отсутствует в выбранной редакции."})
    assert validate_draft(draft, item, fragments, norms, [no_quote]).reason_code == "norm_quote_not_in_unit"


def test_context_quote_cannot_replace_claim_evidence(tmp_path):
    item, fragments, norms, evidence, draft, _ = sample(tmp_path)
    context = fragments[0].spec.model_copy(update={"role": "context"})
    altered = item.model_copy(update={"fragments": [context]})
    loaded = [fragments[0].__class__(context, fragments[0].text, fragments[0].source_path)]
    result = validate_draft(draft, altered, loaded, norms, evidence)
    assert result.reason_code == "doc_quote_not_claim_evidence"


def test_document_path_cannot_escape_notice_root(tmp_path):
    item, _, _, _, _, documents = sample(tmp_path)
    bad = item.model_copy(update={"fragments": [
        DocumentFragment(fragment_id="outside", path="../secret.txt", char_start=0, char_end=5)
    ]})
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_fragments(bad, documents)


def test_quote_from_another_file_or_changed_source_is_rejected(tmp_path):
    item, fragments, norms, evidence, draft, documents = sample(tmp_path)
    (documents / "other.txt").write_text("Иной документ закупки.", encoding="utf-8")
    other = DocumentFragment(
        fragment_id="other", path="other.txt", char_start=0, char_end=len("Иной документ закупки.")
    )
    with_other = item.model_copy(update={"fragments": [*item.fragments, other]})
    loaded = load_fragments(with_other, documents)
    wrong_file_draft = draft.model_copy(update={"source_fragment_id": "other"})
    assert validate_draft(
        wrong_file_draft, with_other, loaded, norms, evidence
    ).reason_code == "doc_span_not_in_fragment"

    (documents / "notice.txt").write_text(DOC_TEXT.replace("Требуется", "Убрано    "), encoding="utf-8")
    assert validate_draft(draft, item, fragments, norms, evidence).reason_code == "source_changed"


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, query):
        return RetrievalEvidence(["art.31"], self.evidence, ["c31"])


class FakeGenerator:
    def __init__(self, draft=None, failure=None):
        self.draft = draft
        self.failure = failure
        self.calls = 0

    def generate(self, argument, fragments, evidence):
        self.calls += 1
        if self.failure:
            raise self.failure
        return self.draft


def test_pipeline_success_and_missing_fragment_refusal(tmp_path):
    item, _, norms, evidence, draft, documents = sample(tmp_path)
    edition = tmp_path / "edition"
    edition.mkdir()
    (edition / "norm_units.jsonl").write_text(
        norms[NORM_ID].model_dump_json() + "\n", encoding="utf-8"
    )
    generator = FakeGenerator(draft)
    pipeline = GroundedArgumentPipeline(
        edition_dir=edition, documents_root=documents,
        retriever=FakeRetriever(evidence), generator=generator,
    )
    result, trace = pipeline.run(item)
    assert result.status == "remark"
    assert trace["retrieval"]["article_ids"] == ["art.31"]
    assert generator.calls == 1

    scope = "Предмет: устройство СКУД."
    shorter = item.model_copy(update={"fragments": [DocumentFragment(
        fragment_id="requirement", path="notice.txt", char_start=0, char_end=len(scope)
    )]})
    result, _ = pipeline.run(shorter)
    assert result.status == "insufficient_evidence"
    assert result.reason_code == "doc_span_not_in_fragment"

    context_only = item.model_copy(update={"fragments": [item.fragments[0].model_copy(
        update={"role": "context"}
    )]})
    result, _ = pipeline.run(context_only)
    assert result.status == "insufficient_evidence"
    assert result.reason_code == "no_claim_fragment"


def test_generation_failure_is_error_not_abstention(tmp_path):
    item, _, norms, evidence, _, documents = sample(tmp_path)
    edition = tmp_path / "edition"
    edition.mkdir()
    (edition / "norm_units.jsonl").write_text(norms[NORM_ID].model_dump_json() + "\n")
    pipeline = GroundedArgumentPipeline(
        edition_dir=edition, documents_root=documents,
        retriever=FakeRetriever(evidence),
        generator=FakeGenerator(failure=GenerationError("offline")),
    )
    result, _ = pipeline.run(item)
    assert result.status == "error"
    assert result.reason_code == "generation_error"


def test_prompt_never_uses_gold_labels(tmp_path):
    item, fragments, _, evidence, _, _ = sample(tmp_path)
    messages = build_messages(item.complaint_argument, fragments, evidence)
    assert "gold" not in json.dumps(messages, ensure_ascii=False).lower()
    assert item.complaint_argument in messages[1]["content"]


def test_schema_allows_only_exposed_ids(tmp_path):
    _, fragments, _, evidence, _, _ = sample(tmp_path)
    fields = draft_schema(fragments, evidence)["$defs"]["RemarkDraft"]["properties"]
    assert fields["source_fragment_id"]["enum"] == ["requirement"]
    assert fields["doc_span_id"]["enum"] == ["s0", "s1"]
    assert fields["norm_chunk_id"]["enum"] == ["c31"]
    assert fields["norm_id"]["enum"] == [NORM_ID]
    assert fields["norm_span_id"]["enum"] == ["s0"]


def test_llama_server_reports_bad_json_and_unavailability(monkeypatch):
    class BadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            return b'{"choices":[{"message":{"content":"not json"}}]}'

    generator = LlamaServerGenerator()
    monkeypatch.setattr("generate.llama_server.request.urlopen", lambda *a, **k: BadResponse())
    with pytest.raises(GenerationError, match="invalid model response"):
        generator.generate("query", [], [])
    monkeypatch.setattr(
        "generate.llama_server.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(URLError("offline"))
    )
    with pytest.raises(GenerationError, match="unavailable"):
        generator.generate("query", [], [])


def test_llama_server_sends_schema_and_parses_draft(monkeypatch):
    captured = {}
    payload = AbstainDraft(decision="abstain", reason="Недостаточно сведений").model_dump_json()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            return json.dumps({"choices": [{"message": {"content": payload}}]}).encode()

    def fake_urlopen(req, timeout):
        captured["request"] = json.loads(req.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("generate.llama_server.request.urlopen", fake_urlopen)
    result = LlamaServerGenerator().generate("query", [], [])
    assert result.decision == "abstain"
    assert captured["request"]["response_format"]["type"] == "json_schema"
    assert captured["request"]["temperature"] == 0
    assert "chat_template_kwargs" not in captured["request"]

    LlamaServerGenerator(
        model="Qwen3-8B", temperature=0.6, max_tokens=4096,
        top_p=0.95, top_k=20, enable_thinking=True, seed=42,
    ).generate("query", [], [])
    assert captured["request"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["request"]["max_tokens"] == 4096
    assert captured["request"]["seed"] == 42


def test_frozen_retriever_is_deterministic_with_fake_sources(tmp_path):
    chunks = [Chunk(
        chunk_id=f"c{i}", edition_id="2026-08-04", chunking="part", text=f"norm {i}",
        char_start=i * 10, char_end=i * 10 + 6, norm_ids=[f"44FZ:2026-08-04:art.{i}:ch.1"],
    ) for i in range(1, 4)]

    class Search:
        def __init__(self, ranked):
            self.chunks = chunks
            self.ranked = ranked

        def search(self, query, k):
            return self.ranked[:k]

    class Scorer:
        def score(self, query, candidates):
            return [float(chunk.chunk_id[1:]) for chunk in candidates]

    retriever = FrozenArticleRetriever(
        Search(chunks), Search(chunks[::-1]), Search(chunks), Scorer(), chunks700=chunks
    )
    first = retriever.retrieve("query")
    second = retriever.retrieve("query")
    assert first == second
    assert first.article_ids
    assert all(chunk.chunk_id in first.scored_chunk_ids for chunk in first.chunks)


def test_frozen_retriever_respects_evidence_chunk_limit():
    chunks = [Chunk(
        chunk_id=f"c{i}", edition_id="2026-08-04", chunking="part", text=f"norm {i}",
        char_start=i * 10, char_end=i * 10 + 6, norm_ids=[f"44FZ:2026-08-04:art.{i}:ch.1"],
    ) for i in range(1, 8)]

    class Search:
        def __init__(self, ranked):
            self.chunks = chunks
            self.ranked = ranked

        def search(self, query, k):
            return self.ranked[:k]

    class Scorer:
        def score(self, query, candidates):
            return [float(chunk.chunk_id[1:]) for chunk in candidates]

    retriever = FrozenArticleRetriever(
        Search(chunks), Search(chunks[::-1]), Search(chunks), Scorer(),
        chunks700=chunks, evidence_max_chunks=5,
    )

    assert len(retriever.retrieve("query").chunks) == 5
