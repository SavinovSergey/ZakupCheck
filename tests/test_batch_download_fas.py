"""Тесты batch-оркестратора без сети."""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "batch_download_fas.py"


def load_batch_module():
    spec = importlib.util.spec_from_file_location("batch_download_fas_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ready_meta(out_dir: Path, number: str) -> None:
    (out_dir / "text").mkdir(parents=True)
    (out_dir / "text" / "card_information.txt").write_text("card\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "complaint_number": number,
                "documents": [],
                "extracted": [
                    {
                        "source": "raw/card_information.html",
                        "text_path": "text/card_information.txt",
                        "method": "html",
                        "chars": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_numbers_validates_and_deduplicates(tmp_path: Path) -> None:
    batch = load_batch_module()
    numbers = tmp_path / "numbers.txt"
    numbers.write_text(
        "123456\n"
        "https://example.test/?complaintNumber=789012\n"
        "123456\n"
        "../escape\n"
        "не номер\n",
        encoding="utf-8",
    )
    assert batch.load_numbers(numbers) == ["123456", "789012"]


def test_ready_requires_existing_html_text(tmp_path: Path) -> None:
    batch = load_batch_module()
    out_dir = tmp_path / "123"
    out_dir.mkdir()
    (out_dir / "meta.json").write_text(
        json.dumps({"extracted": [{"method": "pending", "source": "scan.pdf"}]}),
        encoding="utf-8",
    )
    assert not batch.complaint_ready(out_dir)
    write_ready_meta(out_dir, "123")
    assert batch.complaint_ready(out_dir)


def test_corrupt_meta_is_reported_not_raised(tmp_path: Path) -> None:
    batch = load_batch_module()
    out_dir = tmp_path / "123"
    out_dir.mkdir()
    (out_dir / "meta.json").write_text("{broken", encoding="utf-8")
    summary = batch.summarize_dir(out_dir)
    assert summary["has_meta"] is True
    assert summary["meta_error"]


def test_document_error_is_in_summary(tmp_path: Path) -> None:
    batch = load_batch_module()
    out_dir = tmp_path / "123"
    write_ready_meta(out_dir, "123")
    meta_path = out_dir / "meta.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["documents"] = [
        {"title": "decision.pdf", "status": "error", "error": "timeout"}
    ]
    meta_path.write_text(json.dumps(raw), encoding="utf-8")
    summary = batch.summarize_dir(out_dir)
    assert summary["document_errors"] == ["decision.pdf: timeout"]


def test_successful_batch_has_no_inter_complaint_sleep(tmp_path: Path, monkeypatch) -> None:
    batch = load_batch_module()
    numbers = tmp_path / "numbers.txt"
    numbers.write_text("111\n222\n", encoding="utf-8")
    sleeps: list[float] = []

    async def fake_fetch(number: str, out_root: Path, **_kwargs: object) -> Path:
        out_dir = out_root / number
        write_ready_meta(out_dir, number)
        return out_dir

    async def fake_finish(_tasks: list) -> list[dict]:
        return []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(batch, "fetch_complaint_async", fake_fetch)
    monkeypatch.setattr(batch, "finish_heavy", fake_finish)
    monkeypatch.setattr(batch.asyncio, "sleep", fake_sleep)

    rc = batch.main(
        [
            "--numbers-file",
            str(numbers),
            "--out-root",
            str(tmp_path / "out"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--no-with-notice",
        ]
    )
    assert rc == 0
    assert sleeps == []
    rows = [
        json.loads(line)
        for line in next((tmp_path / "reports").glob("batch_*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["status"] for row in rows] == ["ok", "ok"]


def test_existing_link_failure_is_partial_and_reported(tmp_path: Path, monkeypatch) -> None:
    batch = load_batch_module()
    number = "333"
    numbers = tmp_path / "numbers.txt"
    numbers.write_text(number + "\n", encoding="utf-8")
    out_root = tmp_path / "out"
    write_ready_meta(out_root / number, number)

    async def fake_attach(*_args: object, **_kwargs: object) -> list[Path]:
        raise ValueError("broken notice meta")

    async def fake_finish(_tasks: list) -> list[dict]:
        return []

    monkeypatch.setattr(batch, "attach_notices_async", fake_attach)
    monkeypatch.setattr(batch, "finish_heavy", fake_finish)

    rc = batch.main(
        [
            "--numbers-file",
            str(numbers),
            "--out-root",
            str(out_root),
            "--notice-out-root",
            str(tmp_path / "notices"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--sleep",
            "0",
        ]
    )
    assert rc == 1
    row = json.loads(
        next((tmp_path / "reports").glob("batch_*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["status"] == "partial"
    assert "broken notice meta" in row["error"]


def test_http_404_is_not_retried(monkeypatch) -> None:
    from ingest.fas.download import http_get

    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "https://example.test/missing", 404, "Not Found", hdrs=None, fp=None
        )

    monkeypatch.setattr("ingest.fas.download.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("ingest.fas.download.time.sleep", sleeps.append)

    with pytest.raises(urllib.error.HTTPError) as raised:
        http_get("https://example.test/missing", retries=3)
    assert raised.value.code == 404
    assert calls == 1
    assert sleeps == []
