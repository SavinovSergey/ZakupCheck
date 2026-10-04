"""Скачивание извещения 44-ФЗ и приложений по номеру закупки (regNumber)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingest.fas.download import (
    ComplaintMeta,
    decode_html,
    download_listed_documents,
    extract_all_files,
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
            try:
                data, _ctype, final = http_get(
                    url, timeout=timeout, sleep_s=sleep_s, insecure=insecure, retries=2
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
                return kind, html, final
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}/{key}: {exc}")
            time.sleep(sleep_s)
    raise RuntimeError(
        f"Не удалось открыть извещение {procurement_id}. Пробовали {NOTICE_KINDS}. "
        f"Последние ошибки: {errors[-6:]}"
    )


def fetch_notice(
    procurement_id: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 90.0,
    skip_download: bool = False,
    insecure: bool = False,
    kind: str | None = None,
) -> Path:
    out_dir = out_root / procurement_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if kind:
        urls = notice_urls(procurement_id, kind)
        data, _ctype, final = http_get(
            urls["documents"], timeout=timeout, sleep_s=sleep_s, insecure=insecure
        )
        html = decode_html(data)
        page_key = "documents"
        if len(html) < 1500:
            data, _ctype, final = http_get(
                urls["common"], timeout=timeout, sleep_s=sleep_s, insecure=insecure
            )
            html = decode_html(data)
            page_key = "common"
        notice_kind = kind
    else:
        notice_kind, html, final = probe_notice_kind(
            procurement_id, timeout=timeout, sleep_s=sleep_s, insecure=insecure
        )
        page_key = "probe"
        urls = notice_urls(procurement_id, notice_kind)

    (raw_dir / f"notice_{page_key}.html").write_text(html, encoding="utf-8")

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
        card_url=urls["common"],
        documents_url=urls["documents"],
        linked_complaints=existing_linked,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )

    combined_html = html
    for key in ("documents", "common"):
        dest = raw_dir / f"notice_{key}.html"
        if dest.exists():
            combined_html += "\n" + dest.read_text(encoding="utf-8")
            continue
        try:
            data, _ctype, final_u = http_get(
                urls[key], timeout=timeout, sleep_s=sleep_s, insecure=insecure
            )
            time.sleep(sleep_s)
            page_html = decode_html(data)
            if len(page_html) >= 1500:
                dest.write_text(page_html, encoding="utf-8")
                combined_html += "\n" + page_html
                if key == "common":
                    meta.card_url = final_u
                else:
                    meta.documents_url = final_u
        except Exception as exc:  # noqa: BLE001
            meta.notes.append(f"notice {key} failed: {exc}")

    links = parse_document_links(combined_html)
    meta.documents = [{"url": l.url, "title": l.title, "status": "listed"} for l in links]

    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    for html_file in sorted(raw_dir.glob("notice_*.html")):
        card_text = extract_html(html_file.read_text(encoding="utf-8"))
        out_txt = text_dir / f"{html_file.stem}.txt"
        out_txt.write_text(card_text + "\n", encoding="utf-8")
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
        bridge = download_listed_documents(
            bridge, out_dir, sleep_s=sleep_s, timeout=timeout, insecure=insecure
        )
        meta.documents = bridge.documents
        meta.notes = bridge.notes
    bridge = extract_all_files(bridge, out_dir)
    meta.extracted = bridge.extracted
    meta.notes = bridge.notes

    (out_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_dir
