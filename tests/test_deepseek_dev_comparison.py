from __future__ import annotations

import io
import json

from scripts.run_deepseek_dev_comparison import DeepSeekGenerator, api_key, outgoing_messages
from schemas.corpus import Chunk
from schemas.grounded import DocumentFragment
from validate.grounding import LoadedFragment


def _example():
    fragment = LoadedFragment(
        spec=DocumentFragment(
            fragment_id="claim", role="claim", path="notice/Иванов.txt",
            char_start=0, char_end=100,
        ),
        text="Жалобу ИП Иванова Ивана Ивановича признать обоснованной.\nТребование лицензии на монтаж пожарной сигнализации.",
        source_path=None,
    )
    chunk = Chunk(
        chunk_id="chunk-1", edition_id="2026-08-04", chunking="point",
        text="Не допускается установление требований, влекущих ограничение числа участников закупки.",
        char_start=0, char_end=88,
        norm_ids=["44FZ:2026-08-04:art.42:ch.2:p.3"],
    )
    return fragment, chunk


def test_outbound_prompt_redacts_person_and_path_but_keeps_original_span_ids():
    fragment, chunk = _example()
    messages, counts = outgoing_messages("Довод ИП Иванова Ивана Ивановича", [fragment], [chunk])

    user = messages[1]["content"]
    assert "Иванова Ивана Ивановича" not in user
    assert "notice/Иванов.txt" not in user
    assert "<PERSON_1>" in user
    assert "<LOCAL_SOURCE>" in user
    assert "[s0]" in user
    assert "[s1]" in user
    assert counts["PERSON"] >= 1


def test_generator_sends_json_mode_and_never_sends_raw_person(monkeypatch):
    fragment, chunk = _example()
    seen = {}

    def fake_urlopen(outgoing, timeout):
        seen["url"] = outgoing.full_url
        seen["payload"] = json.loads(outgoing.data)
        response = {
            "model": "deepseek-v4-pro",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"decision":"abstain","reason":"Недостаточно данных"}'}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 20},
        }
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("scripts.run_deepseek_dev_comparison.request.urlopen", fake_urlopen)
    generator = DeepSeekGenerator(
        token="test-token", effort="high", max_tokens=8192,
        timeout_seconds=30, manual_redactions=[],
    )
    draft = generator.generate("Довод ИП Иванова Ивана Ивановича", [fragment], [chunk])

    assert draft.decision == "abstain"
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["reasoning_effort"] == "high"
    assert "Иванова Ивана Ивановича" not in json.dumps(seen["payload"], ensure_ascii=False)
    assert generator.last_generation_metadata["usage"]["prompt_tokens"] == 200


def test_key_reads_local_env_file_without_logging(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY='private-value'\n", encoding="utf-8")
    assert api_key(env_file) == "private-value"
