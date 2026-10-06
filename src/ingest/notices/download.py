"""Скачивание извещения 44-ФЗ и приложений по номеру закупки (regNumber)."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingest.fas.download import (
    ComplaintMeta,
    FORBIDDEN_COOLDOWN_S,
    _is_forbidden,
    decode_html,
    download_listed_documents,
    download_url,
    extract_all_files_async,
    finish_heavy,
    http_get,
    parse_document_links,
)
from ingest.fas.extract import extract_html

# Типы извещений в URL ЕИС (пробуем по очереди, пока страница не откроется).
NOTICE_KINDS = (
    "ea20",
    "ea44",
    "ok20",
    "ok44",
    "zk20",
    "zk44",
    "ep44",
    "ezt20",
    "ezt44",
)


@dataclass
class NoticeMeta:
    procurement_id: str
    notice_kind: str | None = None
    card_url: str | None = None
    documents_url: str | None = None
    source: str = "eis"
    linked_complaints: list[str] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    extracted: list[dict] = field(default_factory=list)
    downloaded_at: str | None = None
    notes: list[str] = field(default_factory=list)


def notice_urls(procurement_id: str, kind: str) -> dict[str, str]:
    base = f"https://zakupki.gov.ru/epz/order/notice/{kind}/view"
    q = f"regNumber={procurement_id}"
    return {
        "common": f"{base}/common-info.html?{q}",
        "documents": f"{base}/documents.html?{q}",
    }


def probe_notice_kind(
    procurement_id: str,
    *,
    timeout: float,
    sleep_s: float,
    insecure: bool,
) -> tuple[str, str, str]:
    """Возвращает (kind, html, url)."""
    errors: list[str] = []
    for kind in NOTICE_KINDS:
        urls = notice_urls(procurement_id, kind)
        for key in ("documents", "common"):
            url = urls[key]
            print(f"probe notice {procurement_id}: try {kind}/{key} …", file=sys.stderr)
            try:
                data, _ctype, final = http_get(
                    url, timeout=timeout, sleep_s=sleep_s, insecure=insecure, retries=1
                )
                html = decode_html(data)
                if len(html) < 1500:
                    errors.append(f"{kind}/{key}: short {len(html)}")
                    continue
                low = html.lower()[:4000]
                if "не найден" in low and ("404" in html[:3000] or "ошибка" in low):
                    errors.append(f"{kind}/{key}: soft 404")
                    continue
                if procurement_id not in html and "regNumber" not in html:
                    errors.append(f"{kind}/{key}: no regNumber marker")
                    continue
                print(f"probe notice {procurement_id}: ok {kind}/{key}", file=sys.stderr)
                return kind, html, final
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}/{key}: {exc}")
    raise RuntimeError(
        f"Не удалось открыть извещение {procurement_id}. Пробовали {NOTICE_KINDS}. "
        f"Последние ошибки: {errors[-6:]}"
    )


def _page_ok(html: str, procurement_id: str) -> bool:
    if len(html) < 1500:
        return False
    low = html.lower()[:4000]
    if "не найден" in low and ("404" in html[:3000] or "ошибка" in low):
        return False
    if procurement_id not in html and "regNumber" not in html:
        return False
    return True


async def fetch_notice_async(
    procurement_id: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    skip_download: bool = False,
    insecure: bool = False,
    kind: str | None = None,
    heavy_tasks: list | None = None,
) -> Path:
    """Открыть извещение. По умолчанию сразу ea20, без повторного скачивания той же вкладки."""
    own_tasks = heavy_tasks is None
    tasks: list = [] if own_tasks else heavy_tasks
    try:
        return await _fetch_notice_body(
            procurement_id,
            out_root,
            tasks,
            sleep_s=sleep_s,
            timeout=timeout,
            skip_download=skip_download,
            insecure=insecure,
            kind=kind,
        )
    finally:
        if own_tasks:
            await finish_heavy(tasks)


async def _fetch_notice_body(
    procurement_id: str,
    out_root: Path,
    tasks: list,
    *,
    sleep_s: float,
    timeout: float,
    skip_download: bool,
    insecure: bool,
    kind: str | None,
) -> Path:
    out_dir = out_root / procurement_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    kinds = (kind,) if kind else NOTICE_KINDS
    errors: list[str] = []
    forbidden = False
    notice_kind: str | None = None
    urls: dict[str, str] | None = None
    pages: dict[str, str] = {}
    finals: dict[str, str] = {}

    for candidate in kinds:
        candidate_urls = notice_urls(procurement_id, candidate)
        got = False
        for key in ("documents", "common"):
            dest = raw_dir / f"notice_{key}.html"
            if dest.is_file() and dest.stat().st_size >= 1500 and candidate == kinds[0]:
                saved = await asyncio.to_thread(dest.read_text, encoding="utf-8")
                if _page_ok(saved, procurement_id):
                    pages[key] = saved
                    got = True
                    continue
            try:
                data, _ctype, final = await download_url(
                    candidate_urls[key],
                    timeout=timeout,
                    sleep_s=sleep_s,
                    insecure=insecure,
                    retries=3 if candidate == kinds[0] else 1,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate}/{key}: {exc}")
                if _is_forbidden(exc):
                    forbidden = True
                    break
                continue
            html = await asyncio.to_thread(decode_html, data)
            if not _page_ok(html, procurement_id):
                errors.append(f"{candidate}/{key}: short or not a card")
                continue
            await asyncio.to_thread(dest.write_text, html, encoding="utf-8")
            pages[key] = html
            finals[key] = final
            got = True
        if forbidden:
            break
        if got:
            notice_kind = candidate
            urls = candidate_urls
            break

    if notice_kind is None and forbidden:
        retry_urls = notice_urls(procurement_id, kinds[0])
        for pause in FORBIDDEN_COOLDOWN_S:
            print(
                f"403 у извещения {procurement_id}: пауза {pause:.0f} с и повтор",
                file=sys.stderr,
            )
            await asyncio.sleep(pause)
            try:
                data, _ctype, final = await download_url(
                    retry_urls["documents"],
                    timeout=timeout,
                    sleep_s=sleep_s,
                    insecure=insecure,
                    retries=1,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kinds[0]}/documents: {exc}")
                if not _is_forbidden(exc):
                    break
                continue
            html = await asyncio.to_thread(decode_html, data)
            if not _page_ok(html, procurement_id):
                errors.append(f"{kinds[0]}/documents: short or not a card")
                break
            dest = raw_dir / "notice_documents.html"
            await asyncio.to_thread(dest.write_text, html, encoding="utf-8")
            pages["documents"] = html
            finals["documents"] = final
            notice_kind = kinds[0]
            urls = retry_urls
            break

    if notice_kind is None or urls is None:
        raise RuntimeError(
            f"Не удалось открыть извещение {procurement_id}. Пробовали {kinds}. "
            f"Последние ошибки: {errors[-6:]}"
        )

    existing_linked: list[str] = []
    existing_meta = out_dir / "meta.json"
    if existing_meta.exists():
        try:
            existing_linked = list(
                json.loads(existing_meta.read_text(encoding="utf-8")).get("linked_complaints") or []
            )
        except (OSError, json.JSONDecodeError):
            pass

    meta = NoticeMeta(
        procurement_id=procurement_id,
        notice_kind=notice_kind,
        card_url=finals.get("common") or urls["common"],
        documents_url=finals.get("documents") or urls["documents"],
        linked_complaints=existing_linked,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )
    print(f"notice {procurement_id}: {notice_kind}", file=sys.stderr)

    combined_html = "\n".join(pages[key] for key in ("documents", "common") if key in pages)
    links = await asyncio.to_thread(parse_document_links, combined_html)
    meta.documents = [{"url": link.url, "title": link.title, "status": "listed"} for link in links]

    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    def _card_text(html_file: Path) -> str:
        return extract_html(html_file.read_text(encoding="utf-8"))

    for html_file in sorted(raw_dir.glob("notice_*.html")):
        card_text = await asyncio.to_thread(_card_text, html_file)
        out_txt = text_dir / f"{html_file.stem}.txt"
        await asyncio.to_thread(out_txt.write_text, card_text + "\n", encoding="utf-8")
        meta.extracted.append(
            {
                "source": f"raw/{html_file.name}",
                "text_path": f"text/{out_txt.name}",
                "method": "html",
                "chars": len(card_text),
            }
        )

    bridge = ComplaintMeta(
        complaint_number=procurement_id,
        documents=meta.documents,
        extracted=meta.extracted,
        notes=meta.notes,
        downloaded_at=meta.downloaded_at,
    )
    if not skip_download and bridge.documents:
        bridge = await download_listed_documents(
            bridge,
            out_dir,
            sleep_s=sleep_s,
            timeout=timeout,
            insecure=insecure,
            heavy_tasks=tasks,
        )
        meta.documents = bridge.documents
        meta.notes = bridge.notes
    bridge = await extract_all_files_async(bridge, out_dir, heavy_tasks=tasks)
    meta.extracted = bridge.extracted
    meta.notes = bridge.notes

    (out_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir


def fetch_notice(
    procurement_id: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    skip_download: bool = False,
    insecure: bool = False,
    kind: str | None = None,
) -> Path:
    """Синхронная обёртка: дожидается и скачивания, и OCR/LibreOffice."""
    return asyncio.run(
        fetch_notice_async(
            procurement_id,
            out_root,
            sleep_s=sleep_s,
            timeout=timeout,
            skip_download=skip_download,
            insecure=insecure,
            kind=kind,
        )
    )
