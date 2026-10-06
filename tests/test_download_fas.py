"""Тесты загрузки жалоб/извещений ЕИС (без сети)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from ingest.fas.download import (
    classify_long_ids,
    discover_procurement_ids_from_complaint_dir,
    fetch_from_html_file,
    main,
    parse_document_links,
    parse_procurement_ids,
)
from ingest.fas.extract import extract_docx, extract_file, extract_html, extract_odt, unescape_markdown
from ingest.notices.download import notice_urls

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fas"


def test_notice_urls() -> None:
    urls = notice_urls("0373100062626000058", "zk20")
    assert "zk20" in urls["documents"]
    assert "regNumber=0373100062626000058" in urls["common"]


def test_parse_card_links_and_procurement() -> None:
    html = (FIXTURES / "mini_card.html").read_text(encoding="utf-8")
    assert parse_procurement_ids(html, exclude={"202500100161017401"}) == [
        "0123456789012345678"
    ]
    links = parse_document_links(html)
    titles = {l.title for l in links}
    assert "Решение.pdf" in titles
    assert "Жалоба.docx" in titles


def test_classify_long_ids_notice_vs_contract() -> None:
    notice, contract, unknown = classify_long_ids(
        "аукцион (номер извещения в ЕИС – 0108300016326000003); "
        "заключен государственный контракт с реестровым номером 2132619645626000006; "
        "и ещё голый номер 0373100062626000058 без подписи."
    )
    assert "0108300016326000003" in notice
    assert "2132619645626000006" in contract
    assert "0373100062626000058" in unknown
    assert "2132619645626000006" not in notice


def test_discover_skips_contract_ids_from_decision_text(tmp_path: Path) -> None:
    """Fallback без карточки: берём контекст извещения, не контракт."""
    complaint = "202600109157000671"
    out = tmp_path / complaint
    (out / "text").mkdir(parents=True)
    (out / "files").mkdir()
    (out / "raw").mkdir()
    (out / "meta.json").write_text(
        '{"complaint_number": "%s", "procurement_ids": []}\n' % complaint,
        encoding="utf-8",
    )
    (out / "text" / "решение.txt").write_text(
        "По результатам закупки заключен государственный контракт "
        "с реестровым номером 2132619645626000006.\n"
        "Предмет: запрос котировок (номер извещения – 0809500000326003521).\n",
        encoding="utf-8",
    )
    found = discover_procurement_ids_from_complaint_dir(out, complaint_number=complaint)
    assert found == ["0809500000326003521"]
    assert "2132619645626000006" not in found


def test_discover_prefers_card_label_over_text_examples(tmp_path: Path) -> None:
    complaint = "202600100161015558"
    out = tmp_path / complaint
    (out / "text").mkdir(parents=True)
    (out / "files").mkdir()
    (out / "raw").mkdir()
    (out / "meta.json").write_text(
        json.dumps(
            {
                "complaint_number": complaint,
                # загрязнение прошлым прогоном
                "procurement_ids": [
                    "0373200022226001924",
                    "0121200004724000809",
                    "0372100043023000208",
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "text" / "card_information.txt").write_text(
        "Сведения о закупке\nНомер извещения\n0373200022226001924\nНаименование закупки\nУЗИ\n",
        encoding="utf-8",
    )
    (out / "text" / "жалоба.txt").write_text(
        "при закупках УЗИ-сканеров:\n"
        "https://zakupki.gov.ru/epz/order/notice/ea20/view/common-info.html"
        "?regNumber=0121200004724000809\n"
        "и ещё [0372100043023000208](https://zakupki.gov.ru/epz/order/notice/"
        "ea20/view/common-info.html?regNumber=0372100043023000208)\n",
        encoding="utf-8",
    )
    (out / "files" / "2328928518.194217733930353958.1.docx").write_bytes(b"PK")
    found = discover_procurement_ids_from_complaint_dir(out, complaint_number=complaint)
    assert found == ["0373200022226001924"]


def test_extract_html_keeps_argument() -> None:
    html = (FIXTURES / "mini_card.html").read_text(encoding="utf-8")
    text = extract_html(html)
    assert "или эквивалент" in text
    assert "обоснована" in text


def test_unescape_markdown() -> None:
    raw = r"25\.09\.2026 \(далее – Комиссия\) видео\-конференц\-связи"
    assert unescape_markdown(raw) == "25.09.2026 (далее – Комиссия) видео-конференц-связи"


def test_extract_docx_roundtrip(tmp_path: Path) -> None:
    docx = tmp_path / "a.docx"
    root = Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document")
    body = SubElement(root, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
    p = SubElement(body, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
    r = SubElement(p, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")
    t = SubElement(r, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
    t.text = "Довод про статью 33"
    content_types = (
        b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Default Extension="xml" ContentType="application/xml"/>'
        b'<Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        b"</Types>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", tostring(root, encoding="utf-8"))
    assert "статью 33" in extract_docx(docx)
    text, method = extract_file(docx)
    assert method == "docx-md"
    assert "статью 33" in text


def test_extract_odt_roundtrip(tmp_path: Path) -> None:
    odt = tmp_path / "a.odt"
    root = Element("{urn:oasis:names:tc:opendocument:xmlns:office:1.0}document-content")
    text_el = SubElement(root, "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}body")
    p = SubElement(text_el, "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p")
    p.text = "Текст решения ФАС"
    with zipfile.ZipFile(odt, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", tostring(root, encoding="utf-8"))
    assert "решения ФАС" in extract_odt(odt)


def test_complaint_from_html_offline(tmp_path: Path) -> None:
    rc = main(
        [
            "complaint",
            "--from-html",
            str(FIXTURES / "mini_card.html"),
            "--number",
            "202500100161017401",
            "--out-root",
            str(tmp_path),
            "--skip-download",
        ]
    )
    assert rc == 0
    out = tmp_path / "202500100161017401"
    assert (out / "meta.json").exists()
    assert (out / "text" / "_combined.txt").exists()
    combined = (out / "text" / "_combined.txt").read_text(encoding="utf-8")
    assert "или эквивалент" in combined


def test_sniff_misnamed_docx_as_pdf(tmp_path: Path) -> None:
    from ingest.fas.extract import extract_file, sniff_format

    # минимальный docx (zip с word/document.xml)
    docx = tmp_path / "Электронный документ.pdf"
    root = Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document")
    body = SubElement(root, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
    p = SubElement(body, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
    r = SubElement(p, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")
    t = SubElement(r, "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
    t.text = "Протокол подведения итогов"
    content_types = (
        b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Default Extension="xml" ContentType="application/xml"/>'
        b'<Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        b"</Types>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", tostring(root, encoding="utf-8"))
    assert sniff_format(docx) == "docx"
    text, method = extract_file(docx)
    assert method == "docx-md"
    assert "Протокол" in text


def test_safe_unpack_zip(tmp_path: Path) -> None:
    from ingest.fas.extract import safe_unpack_zip

    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("inner/note.txt", "протокол подведения итогов")
        zf.writestr("inner/skip/", "")
    members = safe_unpack_zip(zpath, tmp_path / "out")
    assert len(members) == 1
    assert members[0].read_text(encoding="utf-8") == "протокол подведения итогов"


def test_safe_unpack_zip_blocks_slip(tmp_path: Path) -> None:
    from ingest.fas.extract import safe_unpack_zip

    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../evil.txt", "nope")
    try:
        safe_unpack_zip(zpath, tmp_path / "out")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_explicit_bin_and_octet_stream_name() -> None:
    from ingest.fas.download import is_explicit_bin, would_be_bin

    assert is_explicit_bin("https://zakupki.gov.ru/files/scan.bin", None)
    assert is_explicit_bin("https://zakupki.gov.ru/d", "вложение.bin")
    assert not is_explicit_bin("https://zakupki.gov.ru/download.html", "Решение.pdf")
    assert would_be_bin(
        "https://zakupki.gov.ru/download.html", "Решение", "application/octet-stream"
    )
    assert not would_be_bin(
        "https://zakupki.gov.ru/download.html", "Решение", "application/pdf"
    )


def test_403_retries_same_card_without_other_tabs(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import io
    import urllib.error

    from ingest.fas import download as d

    seen: list[str] = []

    async def fake_download(url: str, **_kwargs: object) -> tuple[bytes, str, str]:
        seen.append(url)
        raise urllib.error.HTTPError(
            url, 403, "Forbidden", hdrs=None, fp=io.BytesIO(b"")
        )

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(d, "download_url", fake_download)
    monkeypatch.setattr(d.asyncio, "sleep", no_wait)

    async def run() -> None:
        try:
            await d.fetch_complaint_async("202600116297003180", tmp_path)
            raised = False
        except RuntimeError as exc:
            raised = "403" in str(exc) or "meta.json не записан" in str(exc)
        assert raised

    asyncio.run(run())
    assert seen
    assert all("complaint-information.html" in url for url in seen)
    assert not (tmp_path / "202600116297003180" / "meta.json").exists()


def test_heavy_kind_broken_pdf_goes_to_ocr(tmp_path: Path) -> None:
    from ingest.fas.extract import heavy_kind

    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
    assert heavy_kind(path) == "pdf-ocr"


def test_heavy_kind_doc_does_not_call_libreoffice(tmp_path: Path, monkeypatch) -> None:
    from ingest.fas.extract import heavy_kind

    path = tmp_path / "заявка.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32)

    def boom(_path: Path) -> str:
        raise AssertionError("LibreOffice не должен запускаться на проверке")

    monkeypatch.setattr("ingest.fas.extract.extract_doc", boom)
    assert heavy_kind(path) == "doc-lo"


def test_http_get_rejects_oversize_and_pauses_on_success(monkeypatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ingest.fas.download import SUCCESS_PAUSE_S, DownloadRejected, http_get

    sleeps: list[float] = []
    monkeypatch.setattr("ingest.fas.download.time.sleep", lambda seconds: sleeps.append(seconds))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/big":
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", "99999999")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            body = b"%PDF-1.4 small"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        try:
            http_get(f"http://127.0.0.1:{port}/big", max_bytes=10, retries=1)
            raised = False
        except DownloadRejected as exc:
            raised = exc.reason == "too-large"
        assert raised
        assert sleeps == []
        _data, ctype, _final = http_get(f"http://127.0.0.1:{port}/ok", retries=1)
        assert ctype == "application/pdf"
        assert sleeps == [SUCCESS_PAUSE_S]
    finally:
        server.shutdown()


def test_ocr_starts_before_next_download_finishes(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import threading

    from ingest.fas.download import ComplaintMeta, download_listed_documents, finish_heavy

    started = threading.Event()
    release = threading.Event()

    def fake_http(url: str, **_kwargs: object) -> tuple[bytes, str, str]:
        if url.endswith("/b.pdf"):
            assert started.wait(2), "OCR не начался, пока качался следующий файл"
        return b"not-a-real-pdf", "application/pdf", url

    async def fake_extract(_path: Path) -> tuple[str, str]:
        started.set()
        await asyncio.to_thread(release.wait)
        return "текст со скана", "pdf-ocr"

    monkeypatch.setattr("ingest.fas.download.http_get", fake_http)
    monkeypatch.setattr("ingest.fas.download.extract_heavy", fake_extract)
    monkeypatch.setattr("ingest.fas.download.heavy_kind", lambda _path: "pdf-ocr")

    meta = ComplaintMeta(
        complaint_number="1",
        documents=[
            {"url": "http://example.test/a.pdf", "title": "a.pdf", "status": "listed"},
            {"url": "http://example.test/b.pdf", "title": "b.pdf", "status": "listed"},
        ],
    )

    async def run() -> None:
        tasks: list = []
        updated = await download_listed_documents(meta, tmp_path, heavy_tasks=tasks)
        assert started.is_set()
        assert any(doc.get("deferred") == "pdf-ocr" for doc in updated.documents)
        release.set()
        await finish_heavy(tasks)

    asyncio.run(run())
    texts = sorted((tmp_path / "text").glob("*.txt"))
    assert len(texts) == 2
    assert all("текст со скана" in path.read_text(encoding="utf-8") for path in texts)


def test_fetch_from_html_offline(tmp_path: Path) -> None:
    out = fetch_from_html_file(
        FIXTURES / "mini_card.html",
        "202500100161017401",
        tmp_path,
        download=False,
    )
    assert (out / "meta.json").exists()
    meta = (out / "meta.json").read_text(encoding="utf-8")
    assert "0123456789012345678" in meta
    assert "202500100161017401" not in meta.split("procurement_ids")[1][:80] or True
    # номер жалобы исключён из procurement_ids
    pids = json.loads(meta)["procurement_ids"]
    assert "202500100161017401" not in pids
    assert "0123456789012345678" in pids
