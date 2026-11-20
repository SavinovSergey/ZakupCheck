"""Small OpenAI-compatible llama-server adapter for constrained JSON output."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Sequence
from urllib import error, request

from pydantic import TypeAdapter

from schemas.corpus import Chunk
from schemas.grounded import ModelDraft
from validate.grounding import LoadedFragment
from validate.quote_candidates import quote_candidates

_DRAFT_ADAPTER = TypeAdapter(ModelDraft)


class GenerationError(RuntimeError):
    pass


def draft_schema(fragments: Sequence[LoadedFragment], evidence: Sequence[Chunk]) -> dict:
    """Restrict references to IDs that were actually shown in this request."""
    schema = deepcopy(_DRAFT_ADAPTER.json_schema())
    fields = schema["$defs"]["RemarkDraft"]["properties"]
    fields["source_fragment_id"]["enum"] = sorted(
        fragment.spec.fragment_id for fragment in fragments if fragment.spec.role == "claim"
    )
    fields["doc_span_id"]["enum"] = sorted({
        span.span_id for fragment in fragments if fragment.spec.role == "claim"
        for span in quote_candidates(fragment.text)
    })
    fields["norm_chunk_id"]["enum"] = sorted(chunk.chunk_id for chunk in evidence)
    fields["norm_id"]["enum"] = sorted({norm_id for chunk in evidence for norm_id in chunk.norm_ids})
    fields["norm_span_id"]["enum"] = sorted({
        span.span_id for chunk in evidence for span in quote_candidates(chunk.text)
    })
    return schema


def build_messages(
    complaint_argument: str,
    fragments: Sequence[LoadedFragment],
    evidence: Sequence[Chunk],
) -> list[dict[str, str]]:
    source_blocks = "\n\n".join(
        f"Фрагмент {fragment.spec.fragment_id} (роль: {fragment.spec.role}; источник: {fragment.spec.path}):\n"
        + "\n".join(f"[{span.span_id}] {span.text}" for span in quote_candidates(fragment.text))
        for fragment in fragments
    )
    norm_blocks = "\n\n".join(
        f"Чанк {chunk.chunk_id}; допустимые norm_id: {', '.join(chunk.norm_ids)}\n"
        + "\n".join(f"[{span.span_id}] {span.text}" for span in quote_candidates(chunk.text))
        for chunk in evidence
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты помощник эксперта по закупкам. Верни только JSON по схеме. "
                "Довод жалобы — гипотеза, не доказательство; не цитируй его как документ. "
                "Используй только предоставленные фрагменты закупки и нормы. "
                "Роль claim означает оспариваемое условие, роль context — сведения о предмете "
                "закупки. Не считай деятельность, перечисленную в требовании к участнику, "
                "закупаемой работой без подтверждения в context. "
                "Различай требования к участнику и описание объекта закупки: норму об описании "
                "объекта применяй только к условию в самом описании объекта. "
                "Для decision=remark укажи замечание-кандидат и идентификаторы: "
                "source_fragment_id с ролью claim, doc_span_id из него, norm_chunk_id, "
                "norm_id из этого чанка и norm_span_id из него. Для span_id возвращай "
                "только метку без скобок и текста, например s0. Метки [s0], [s1] "
                "обозначают точные отрезки текста; цитаты программа извлечёт сама. "
                "Для decision=abstain верни только reason, если фактов или применимой нормы "
                "недостаточно для замечания. Форматы remark и abstain взаимоисключающие. "
                "Если официальный перечень работ и условие показывают конкретное расхождение, "
                "сформулируй замечание-кандидат с оговоркой о проверке; не требуй окончательного "
                "юридического вывода. Для doc_span_id выбирай отрезок, прямо называющий "
                "оспариваемую деятельность, а не соседний пункт перечня. "
                "Выбирай отрезки, подтверждающие замечание. "
                "remark_text и reason — не более одного предложения каждый. "
                "Не выдавай юридическое заключение и не выдумывай нормы."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Довод:\n{complaint_argument}\n\n"
                f"Фрагменты закупки:\n{source_blocks}\n\n"
                f"Найденные нормы:\n{norm_blocks}"
            ),
        },
    ]


class LlamaServerGenerator:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "T-lite-it-2.1",
        timeout_seconds: int = 240,
        temperature: float = 0,
        max_tokens: int = 768,
        top_p: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.last_generation_metadata: dict | None = None

    def generate(
        self,
        complaint_argument: str,
        fragments: Sequence[LoadedFragment],
        evidence: Sequence[Chunk],
    ) -> ModelDraft:
        self.last_generation_metadata = None
        body = {
            "model": self.model,
            "messages": build_messages(complaint_argument, fragments, evidence),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_remark",
                    "schema": draft_schema(fragments, evidence),
                    "strict": True,
                },
            },
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if self.seed is not None:
            body["seed"] = self.seed
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        endpoint = request.Request(
            self.base_url + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer no-key"},
            method="POST",
        )
        try:
            with request.urlopen(endpoint, timeout=self.timeout_seconds) as response:
                response_body = json.load(response)
            message = response_body["choices"][0]["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise GenerationError("model returned non-text content")
            reasoning = message.get("reasoning_content")
            self.last_generation_metadata = {
                "model": self.model,
                "thinking_requested": self.enable_thinking,
                "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                "usage": response_body.get("usage"),
            }
            return _DRAFT_ADAPTER.validate_json(content)
        except (error.URLError, TimeoutError, OSError) as exc:
            raise GenerationError(f"llama-server unavailable: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise GenerationError(f"invalid model response: {exc}") from exc
