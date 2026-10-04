"""Загрузка актуальной редакции 44-ФЗ с pravo.gov.ru (ИПС «Законодательство России»).

Сохраняет артефакты, удобные для последующего парсинга/чанкинга (§4.2 DESIGN):

  data/law/{edition_id}/
    raw/source.html     — исходный HTML (разметка class=H для заголовков статей)
    raw/card.html       — карточка документа
    raw/docbody.html    — страница со списком редакций
    edition.json        — метаданные Edition + ips_nd/ips_rdk
    full_text.txt       — нормализованный текст (основа для char_start/char_end)
    blocks.jsonl        — линейные блоки (глава/статья/тело) с оффсетами в full_text

Источник (публичный неофициальный HTML-прокси официального банка):
  http://pravo.gov.ru/proxy/ips/?doc_itself=&nd=102164547&rdk=...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

# Документ 44-ФЗ в ИПС «Законодательство России»
DEFAULT_ND = "102164547"
USER_AGENT = "ZakupCheck/0.1 (+local research; respectful crawl)"


@dataclass
class Redaction:
    rdk: int
    label: str
    amending_date: date | None
    amending_law: str | None
    selected: bool = False


@dataclass
class Block:
    order: int
    block_id: str | None
    kind: str  # chapter_title | article_title | body | other
    text: str
    article_hint: int | None
    article_title_hint: str | None
    char_start: int
    char_end: int


class _BlockParser(HTMLParser):
    """Режет HTML закона на блоки по <p> / заголовкам class=H."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self._skip = 0
        self._in_p = False
        self._p_attrs: dict[str, str] = {}
        self._buf: list[str] = []
        self._pending_w9 = False  # Статья 24<span class="W9">1</span> → 24.1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "br" and self._in_p:
            self._buf.append("\n")
            self._pending_w9 = False
        if tag == "span" and self._in_p:
            classes = set((attr.get("class") or "").split())
            # В ИПС дробный номер статьи: 24 + <span class="W9">1</span> = 24.1
            if "W9" in classes:
                self._pending_w9 = True
                return
        if tag == "p":
            self._flush_p()
            self._in_p = True
            self._p_attrs = attr
            self._buf = []
            self._pending_w9 = False

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag == "span":
            self._pending_w9 = False
        if tag == "p" and self._in_p:
            self._flush_p()

    def handle_data(self, data: str) -> None:
        if self._skip or not self._in_p:
            return
        if self._pending_w9:
            # вставляем точку перед суффиксом номера статьи, если её ещё нет
            if self._buf and re.search(r"\d$", self._buf[-1]) and data and data[0].isdigit():
                self._buf.append(".")
            self._pending_w9 = False
        self._buf.append(data)

    def _flush_p(self) -> None:
        if not self._in_p:
            return
        text = normalize_text("".join(self._buf))
        self._in_p = False
        self._buf = []
        if not text:
            return
        classes = set((self._p_attrs.get("class") or "").split())
        kind = "other"
        article_hint = None
        article_title_hint = None
        if "H" in classes:
            if re.match(r"^Глава\s+\d+", text):
                kind = "chapter_title"
            elif m := re.match(r"^Статья\s+(\d+(?:\.\d+)?)\.?\s*(.*)$", text):
                kind = "article_title"
                num = m.group(1)
                # "24.1" → hint 24 (мажор); полный номер остаётся в text
                article_hint = int(num.split(".")[0])
                article_title_hint = m.group(2).strip() or None
                # нормализуем заголовок к виду «Статья 24.1. …»
                rest = m.group(2).strip()
                text = f"Статья {num}." + (f" {rest}" if rest else "")
            else:
                kind = "chapter_title"
        else:
            kind = "body"
        self.blocks.append(
            {
                "block_id": self._p_attrs.get("id") or None,
                "kind": kind,
                "text": text,
                "article_hint": article_hint,
                "article_title_hint": article_title_hint,
            }
        )


_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_NBSP = "\u00a0"


def normalize_text(text: str) -> str:
    """Стабильная нормализация до фиксации оффсетов (DESIGN §4.2)."""
    text = text.replace(_NBSP, " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text).strip()
    return text


def http_get(url: str, *, timeout: float = 90.0, retries: int = 3) -> tuple[bytes, str]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                charset = resp.headers.get_content_charset() or "windows-1251"
                return data, charset
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 8))
    assert last_err is not None
    raise last_err


def decode_html(data: bytes, charset: str) -> str:
    for enc in (charset, "windows-1251", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_redactions(docbody_html: str) -> list[Redaction]:
    """Парсит <select name=doc_editions> со страницы документа."""
    # option value='157,102164547' selected ...>157 - от 04.08.2026 № 330-ФЗ (изм.)
    pattern = re.compile(
        r"<option[^>]*value=['\"](\d+)(?:,\d+)?['\"][^>]*>([^<]*)</option>",
        flags=re.I,
    )
    selected_m = re.search(
        r"<option[^>]*value=['\"](\d+)(?:,\d+)?['\"][^>]*selected[^>]*>",
        docbody_html,
        flags=re.I,
    )
    selected_rdk = int(selected_m.group(1)) if selected_m else None

    out: list[Redaction] = []
    seen: set[int] = set()
    for m in pattern.finditer(docbody_html):
        rdk = int(m.group(1))
        if rdk in seen:
            continue
        seen.add(rdk)
        label = normalize_text(m.group(2))
        amending_date = None
        amending_law = None
        dm = re.search(r"от\s+(\d{2})\.(\d{2})\.(\d{4})", label)
        if dm:
            amending_date = date(int(dm.group(3)), int(dm.group(2)), int(dm.group(1)))
        lm = re.search(r"№\s*([\d\-]+-ФЗ)", label)
        if lm:
            amending_law = lm.group(1)
        out.append(
            Redaction(
                rdk=rdk,
                label=label,
                amending_date=amending_date,
                amending_law=amending_law,
                selected=rdk == selected_rdk,
            )
        )
    out.sort(key=lambda r: r.rdk)
    if selected_rdk is None and out:
        out[-1].selected = True
    return out


def choose_redaction(redactions: list[Redaction], rdk: int | None) -> Redaction:
    if not redactions:
        raise RuntimeError("Не найден список редакций на странице документа")
    if rdk is not None:
        for r in redactions:
            if r.rdk == rdk:
                return r
        raise RuntimeError(f"rdk={rdk} нет в списке редакций")
    for r in redactions:
        if r.selected:
            return r
    return redactions[-1]


def edition_id_for(redaction: Redaction) -> str:
    if redaction.amending_date:
        return redaction.amending_date.isoformat()
    return f"rdk-{redaction.rdk}"


def html_to_blocks(source_html: str) -> tuple[str, list[Block]]:
    parser = _BlockParser()
    parser.feed(source_html)
    parser.close()

    texts = [raw["text"] for raw in parser.blocks]
    full_text = "\n\n".join(texts)

    blocks: list[Block] = []
    cursor = 0
    for i, raw in enumerate(parser.blocks):
        text = raw["text"]
        start = cursor
        end = start + len(text)
        if full_text[start:end] != text:
            raise AssertionError(f"offset mismatch at block {i}: {text[:40]!r}")
        blocks.append(
            Block(
                order=i,
                block_id=raw["block_id"],
                kind=raw["kind"],
                text=text,
                article_hint=raw["article_hint"],
                article_title_hint=raw["article_title_hint"],
                char_start=start,
                char_end=end,
            )
        )
        cursor = end + (2 if i < len(texts) - 1 else 0)
    return full_text, blocks


def build_urls(nd: str, rdk: int) -> dict[str, str]:
    # без fulltext=1 ИПС отдаёт урезанный фрагмент (~до ст.48 для 44-ФЗ)
    doc_url = (
        f"http://pravo.gov.ru/proxy/ips/?doc_itself=&fulltext=1"
        f"&nd={nd}&page=1&rdk={rdk}&link_id=0"
    )
    card_url = (
        f"http://pravo.gov.ru/proxy/ips/?doc_itself=&vkart=card&nd={nd}"
        f"&page=1&rdk={rdk}&intelsearch=&link_id=0"
    )
    body_url = f"http://pravo.gov.ru/proxy/ips/?docbody=&nd={nd}"
    return {"doc": doc_url, "card": card_url, "body": body_url}


def download_edition(
    *,
    out_root: Path,
    nd: str = DEFAULT_ND,
    rdk: int | None = None,
    sleep_s: float = 1.0,
) -> Path:
    body_url = f"http://pravo.gov.ru/proxy/ips/?docbody=&nd={nd}"
    print(f"Fetching redaction list: {body_url}", file=sys.stderr)
    body_bytes, body_cs = http_get(body_url)
    body_html = decode_html(body_bytes, body_cs)
    redactions = parse_redactions(body_html)
    chosen = choose_redaction(redactions, rdk)
    edition_id = edition_id_for(chosen)
    urls = build_urls(nd, chosen.rdk)

    print(
        f"Chosen rdk={chosen.rdk} edition_id={edition_id} ({chosen.label})",
        file=sys.stderr,
    )
    time.sleep(sleep_s)

    print(f"Fetching text: {urls['doc']}", file=sys.stderr)
    doc_bytes, doc_cs = http_get(urls["doc"], timeout=120.0)
    doc_html = decode_html(doc_bytes, doc_cs)
    time.sleep(sleep_s)

    print(f"Fetching card: {urls['card']}", file=sys.stderr)
    card_bytes, card_cs = http_get(urls["card"])
    card_html = decode_html(card_bytes, card_cs)

    full_text, blocks = html_to_blocks(doc_html)
    if not full_text.strip():
        raise RuntimeError("После разбора HTML получился пустой full_text")

    article_titles = sum(1 for b in blocks if b.kind == "article_title")
    article_nums = sorted(
        {
            b.article_hint
            for b in blocks
            if b.kind == "article_title" and b.article_hint is not None
        }
    )
    max_article = article_nums[-1] if article_nums else 0
    print(
        f"Parsed blocks={len(blocks)} article_titles={article_titles} "
        f"max_article={max_article} chars={len(full_text)}",
        file=sys.stderr,
    )
    # 44-ФЗ в актуальной редакции идёт примерно до ст.112–114; без fulltext
    # раньше обрывалось около ст.48.
    if nd == DEFAULT_ND and max_article < 90:
        raise RuntimeError(
            f"Похоже, скачан неполный текст 44-ФЗ (max article={max_article}). "
            "Проверьте параметр fulltext=1 в URL ИПС."
        )

    digest = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    out_dir = out_root / edition_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "docbody.html").write_text(body_html, encoding="utf-8")
    (raw_dir / "source.html").write_text(doc_html, encoding="utf-8")
    (raw_dir / "card.html").write_text(card_html, encoding="utf-8")
    (out_dir / "full_text.txt").write_text(full_text + "\n", encoding="utf-8")

    with (out_dir / "blocks.jsonl").open("w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

    edition = {
        "law_id": "44FZ",
        "edition_id": edition_id,
        "effective_from": edition_id if re.fullmatch(r"\d{4}-\d{2}-\d{2}", edition_id) else None,
        "effective_to": None,
        "source_url": urls["doc"],
        "content_sha256": digest,
        "title": (
            "Федеральный закон от 05.04.2013 № 44-ФЗ "
            "«О контрактной системе в сфере закупок…»"
        ),
        "ips_nd": nd,
        "ips_rdk": chosen.rdk,
        "ips_label": chosen.label,
        "ips_amending_law": chosen.amending_law,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "blocks": len(blocks),
            "article_titles": article_titles,
            "max_article": max_article,
            "chars": len(full_text),
        },
        "redactions_available": [
            {
                "rdk": r.rdk,
                "label": r.label,
                "amending_date": r.amending_date.isoformat() if r.amending_date else None,
                "selected": r.rdk == chosen.rdk,
            }
            for r in redactions
            if r.rdk >= max(0, chosen.rdk - 30)  # хвост списка рядом с выбранной
        ],
    }
    (out_dir / "edition.json").write_text(
        json.dumps(edition, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # краткий индекс редакций целиком — для выбора старых окон после MVP
    (out_dir / "raw" / "redactions.json").write_text(
        json.dumps(
            [
                {
                    "rdk": r.rdk,
                    "label": r.label,
                    "amending_date": r.amending_date.isoformat() if r.amending_date else None,
                    "amending_law": r.amending_law,
                    "selected": r.rdk == chosen.rdk,
                }
                for r in redactions
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Скачать редакцию 44-ФЗ с pravo.gov.ru (ИПС) в формате для парсинга"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/law"),
        help="Каталог data/law (по умолчанию ./data/law)",
    )
    parser.add_argument("--nd", default=DEFAULT_ND, help="Идентификатор документа ИПС")
    parser.add_argument(
        "--rdk",
        type=int,
        default=None,
        help="Номер редакции ИПС (по умолчанию — selected/latest)",
    )
    parser.add_argument(
        "--list-redactions",
        action="store_true",
        help="Только показать доступные редакции и выйти",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Пауза между запросами, с")
    args = parser.parse_args(argv)

    if args.list_redactions:
        body_bytes, body_cs = http_get(f"http://pravo.gov.ru/proxy/ips/?docbody=&nd={args.nd}")
        redactions = parse_redactions(decode_html(body_bytes, body_cs))
        for r in redactions:
            mark = "*" if r.selected else " "
            print(f"{mark} rdk={r.rdk:>3}  {r.label}")
        return 0

    out_dir = download_edition(
        out_root=args.out_root,
        nd=args.nd,
        rdk=args.rdk,
        sleep_s=args.sleep,
    )
    print(f"OK: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
