"""Скачивание жалоб ФАС / карточек ЕИС + сырой текст (DESIGN §4.5).

Не размечает EvalCase — только кэш файлов и извлечённый текст для ручной разметки.

  data/raw/fas/{complaint_number}/     — жалоба
  data/raw/notices/{procurement_id}/   — извещение (подкоманда notice)

Поиск по произвольному query убран: номера жалобы/извещения задаются явно.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from ingest.fas.extract import (
    correct_suffix,
    extract_file,
    extract_html,
    normalize_text,
    safe_unpack_zip,
)

USER_AGENT = "ZakupCheck/0.1 (+local research; respectful crawl; FAS eval ingest)"
EIS_BASE = "https://zakupki.gov.ru"

COMPLAINT_NUM_RE = re.compile(r"complaintNumber=(\d+)", re.I)
COMPLAINT_ID_RE = re.compile(r"complaintId=(\d+)", re.I)
REG_NUMBER_RE = re.compile(
    r"(?:regNumber|reestrNumber|purchaseNumber|orderNumber)=(\d{18,19})",
    re.I,
)
ORDER_NOTICE_RE = re.compile(
    r"/epz/order/notice/[^\"'\s>]+[?&]regNumber=(\d{18,19})",
    re.I,
)
# частые ссылки на скачивание вложений ЕИС
DOWNLOAD_HREF_RE = re.compile(
    r'href=["\']([^"\']*(?:downloadDocument|downloadHtml|file\.html|getDocs)[^"\']*)["\']',
    re.I,
)
HREF_ANY_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
TITLE_RE = re.compile(r'title=["\']([^"\']+)["\']', re.I)


@dataclass
class DocLink:
    url: str
    title: str | None = None


@dataclass
class ComplaintMeta:
    complaint_number: str
    complaint_id: str | None = None
    card_url: str | None = None
    source: str = "eis"
    procurement_ids: list[str] = field(default_factory=list)
    notice_dirs: list[str] = field(default_factory=list)  # пути к data/raw/notices/{id}
    documents: list[dict] = field(default_factory=list)
    extracted: list[dict] = field(default_factory=list)
    downloaded_at: str | None = None
    notes: list[str] = field(default_factory=list)


def http_get(
    url: str,
    *,
    timeout: float = 90.0,
    retries: int = 3,
    sleep_s: float = 1.0,
    insecure: bool = False,
) -> tuple[bytes, str, str]:
    """Возвращает (body, content_type, final_url)."""
    last_err: Exception | None = None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }
    ctx: ssl.SSLContext | None = None
    if insecure:
        ctx = ssl._create_unverified_context()  # noqa: S323 — явный флаг --insecure
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = resp.read()
                ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                final = resp.geturl()
                return data, ctype, final
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(max(sleep_s, min(2**attempt, 10)))
    assert last_err is not None
    raise last_err


def decode_html(data: bytes) -> str:
    for enc in ("utf-8", "windows-1251", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def absolute_url(href: str, base: str = EIS_BASE) -> str:
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    return urllib.parse.urljoin(base + "/", href)


def card_urls(complaint_number: str) -> dict[str, str]:
    """Страницы карточки жалобы ЕИС.

    Актуальный UI: complaint-information.html.
    common/documents — устаревшие пути (сейчас стабильно 404), оставляем
    только как запасной порядок, если information недоступна.
    """
    q = urllib.parse.urlencode({"complaintNumber": complaint_number})
    return {
        "information": f"{EIS_BASE}/epz/complaint/card/complaint-information.html?{q}",
        "common": f"{EIS_BASE}/epz/complaint/card/common-info.html?{q}",
        "documents": f"{EIS_BASE}/epz/complaint/card/documents-info.html?{q}",
    }


def parse_procurement_ids(html: str, *, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    # сначала явные ссылки на извещение — надёжнее
    found = ORDER_NOTICE_RE.findall(html) + REG_NUMBER_RE.findall(html)
    return normalize_procurement_ids(found, exclude=exclude)


def normalize_procurement_ids(
    ids: list[str],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Убрать номер жалобы и похожие на жалобу id (20xxxxxxxx…)."""
    exclude = exclude or set()
    out: list[str] = []
    seen: set[str] = set()
    for x in ids:
        if not x or x in exclude or x in seen:
            continue
        # номера жалоб ЕИС обычно 18 цифр и начинаются с 20YY…
        if re.fullmatch(r"20\d{16}", x):
            continue
        seen.add(x)
        out.append(x)
    return out


def discover_procurement_ids_from_complaint_dir(
    out_dir: Path,
    *,
    complaint_number: str,
) -> list[str]:
    """Добрать номер извещения из meta / имён файлов / текста жалобы."""
    found: list[str] = []
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        found.extend(raw.get("procurement_ids") or [])
        found.extend(
            Path(p).name
            for p in (raw.get("notice_dirs") or [])
            if Path(p).name
        )
    for pattern in ("files/*", "text/*"):
        for p in out_dir.glob(pattern):
            found.extend(re.findall(r"\b(\d{18,19})\b", p.name))
            if p.suffix.lower() == ".txt":
                try:
                    found.extend(re.findall(r"\b(\d{18,19})\b", p.read_text(encoding="utf-8")[:8000]))
                except OSError:
                    pass
    return normalize_procurement_ids(found, exclude={complaint_number})


def attach_notices(
    complaint_dir: Path,
    notice_out_root: Path,
    *,
    fetch_missing: bool = True,
    sleep_s: float = 1.5,
    timeout: float = 90.0,
    insecure: bool = False,
) -> list[Path]:
    """Связать жалобу с извещениями: использовать уже скачанные или докачать.

    Пишет в meta жалобы: procurement_ids (очищенные) и notice_dirs.
    """
    meta_path = complaint_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"нет meta.json в {complaint_dir}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    complaint_number = raw.get("complaint_number") or complaint_dir.name
    pids = discover_procurement_ids_from_complaint_dir(
        complaint_dir, complaint_number=complaint_number
    )
    if not pids:
        raise RuntimeError(
            f"Не найден номер извещения для жалобы {complaint_number}. "
            f"Укажите вручную: notice --number <regNumber>, затем "
            f"link --complaint-dir {complaint_dir} --notice-number <regNumber>"
        )

    notice_dirs: list[str] = []
    linked: list[Path] = []
    notice_out_root = Path(notice_out_root)
    notice_out_root.mkdir(parents=True, exist_ok=True)

    from ingest.notices.download import fetch_notice

    for pid in pids:
        npath = notice_out_root / pid
        if (npath / "meta.json").exists():
            print(f"Link existing notice {pid} → {npath}", file=sys.stderr)
        elif fetch_missing:
            print(f"Fetching linked notice {pid} …", file=sys.stderr)
            npath = fetch_notice(
                pid,
                notice_out_root,
                sleep_s=sleep_s,
                timeout=timeout,
                insecure=insecure,
            )
        else:
            print(
                f"skip missing notice {pid} (нет {npath}); "
                f"скачайте: notice --number {pid}",
                file=sys.stderr,
            )
            continue

        # обратная ссылка в meta извещения
        nmeta_path = npath / "meta.json"
        nmeta = json.loads(nmeta_path.read_text(encoding="utf-8"))
        linked_c = list(nmeta.get("linked_complaints") or [])
        if complaint_number not in linked_c:
            linked_c.append(complaint_number)
            nmeta["linked_complaints"] = linked_c
            nmeta_path.write_text(
                json.dumps(nmeta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        rel = str(npath)
        notice_dirs.append(rel)
        linked.append(npath)

    raw["procurement_ids"] = pids
    raw["notice_dirs"] = notice_dirs
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return linked


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[DocLink] = []
        self._href: str | None = None
        self._title: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = {k: (v or "") for k, v in attrs}
        href = attr.get("href") or ""
        if not href or href.startswith("#"):
            return
        self._href = href
        self._title = attr.get("title") or None
        self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        text = normalize_text("".join(self._buf)) or None
        title = self._title or text
        href_l = self._href.lower()
        interesting = any(
            key in href_l
            for key in (
                "downloaddocument",
                "downloadhtml",
                "file.html",
                "getdocs",
                "filestore",
                ".pdf",
                ".docx",
                ".doc",
                ".odt",
                ".rtf",
                ".zip",
            )
        ) or (title and any(x in title.lower() for x in ("решен", "жалоб", "предписан", ".pdf", ".docx")))
        # не скачивать ссылки «перейти в реестр жалоб» / поиск
        junk = any(
            x in href_l
            for x in ("/epz/complaint/search", "searchstring=", "results.html")
        )
        if interesting and not junk:
            self.links.append(DocLink(url=absolute_url(self._href), title=title))
        self._href = None
        self._title = None
        self._buf = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._buf.append(data)


def parse_document_links(html: str) -> list[DocLink]:
    parser = _AnchorCollector()
    parser.feed(html)
    parser.close()
    # dedupe by url
    out: list[DocLink] = []
    seen: set[str] = set()
    for link in parser.links:
        if link.url in seen:
            continue
        seen.add(link.url)
        out.append(link)
    # regex fallback
    for m in DOWNLOAD_HREF_RE.finditer(html):
        url = absolute_url(m.group(1))
        if url not in seen:
            seen.add(url)
            out.append(DocLink(url=url))
    return out


def guess_filename(url: str, title: str | None, content_type: str, index: int) -> str:
    path = urllib.parse.urlparse(url).path
    base = Path(path).name
    if base and "." in base and not base.endswith(".html"):
        return _safe_name(base)
    # from title
    if title:
        t = title.strip()
        for ext in (".pdf", ".docx", ".doc", ".odt", ".rtf", ".zip", ".html"):
            if t.lower().endswith(ext):
                return _safe_name(t)
        # content-type hint
        ext = _ext_from_ctype(content_type)
        if ext:
            return _safe_name(f"{t}{ext}")
    ext = _ext_from_ctype(content_type) or ".bin"
    return f"attachment_{index:02d}{ext}"


def _ext_from_ctype(ctype: str) -> str:
    ctype = (ctype or "").lower()
    mapping = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.oasis.opendocument.text": ".odt",
        "application/rtf": ".rtf",
        "text/rtf": ".rtf",
        "text/html": ".html",
        "text/plain": ".txt",
        "application/zip": ".zip",
    }
    return mapping.get(ctype, "")


def _safe_name(name: str) -> str:
    name = name.replace("\x00", "").strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", " ", name)
    if len(name) > 180:
        stem = Path(name).stem[:150]
        suffix = Path(name).suffix
        name = stem + suffix
    return name or "file.bin"


def complaint_dir(out_root: Path, complaint_number: str) -> Path:
    return out_root / complaint_number


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_card_html(
    *,
    complaint_number: str,
    html: str,
    out_dir: Path,
    card_name: str = "card_common.html",
    card_url: str | None = None,
) -> ComplaintMeta:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / card_name).write_text(html, encoding="utf-8")

    meta = ComplaintMeta(
        complaint_number=complaint_number,
        card_url=card_url,
        procurement_ids=parse_procurement_ids(html, exclude={complaint_number}),
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )
    # id from html if present
    m = COMPLAINT_ID_RE.search(html)
    if m:
        meta.complaint_id = m.group(1)

    links = parse_document_links(html)
    meta.documents = [{"url": l.url, "title": l.title, "status": "listed"} for l in links]

    # plaintext of card itself
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    card_text = extract_html(html)
    (text_dir / f"{Path(card_name).stem}.txt").write_text(card_text + "\n", encoding="utf-8")
    meta.extracted.append(
        {
            "source": f"raw/{card_name}",
            "text_path": f"text/{Path(card_name).stem}.txt",
            "method": "html",
            "chars": len(card_text),
        }
    )
    return meta


def download_listed_documents(
    meta: ComplaintMeta,
    out_dir: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 120.0,
    insecure: bool = False,
) -> ComplaintMeta:
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict] = []
    for i, doc in enumerate(meta.documents, start=1):
        url = doc["url"]
        title = doc.get("title")
        try:
            data, ctype, final_url = http_get(
                url, timeout=timeout, sleep_s=sleep_s, insecure=insecure
            )
            time.sleep(sleep_s)
            fname = guess_filename(final_url or url, title, ctype, i)
            # avoid overwrite
            dest = files_dir / fname
            if dest.exists():
                dest = files_dir / f"{dest.stem}_{i}{dest.suffix}"
            save_bytes(dest, data)
            # ЕИС часто кладёт DOCX/ZIP под именем .pdf — поправим расширение
            fixed = correct_suffix(dest)
            if fixed != dest:
                meta.notes.append(f"renamed by magic: {dest.name} → {fixed.name}")
                dest = fixed
            updated.append(
                {
                    **doc,
                    "status": "downloaded",
                    "path": f"files/{dest.name}",
                    "content_type": ctype,
                    "sha256": sha256_bytes(data),
                    "bytes": len(data),
                    "final_url": final_url,
                }
            )
        except Exception as exc:  # noqa: BLE001 — копим ошибки по файлам
            updated.append({**doc, "status": "error", "error": str(exc)})
            meta.notes.append(f"download failed: {url}: {exc}")
    meta.documents = updated
    return meta


def extract_all_files(meta: ComplaintMeta, out_dir: Path) -> ComplaintMeta:
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    extracted = [e for e in meta.extracted if e.get("method") == "html"]
    combined_parts: list[str] = []

    # include card texts already written
    for e in extracted:
        p = out_dir / e["text_path"]
        if p.exists():
            combined_parts.append(p.read_text(encoding="utf-8"))

    # очередь: (abs_path, source_label, text_rel_stem)
    queue: list[tuple[Path, str, str]] = []
    for doc in meta.documents:
        if doc.get("status") != "downloaded":
            continue
        rel = doc.get("path")
        if not rel:
            continue
        path = out_dir / rel
        queue.append((path, rel, Path(rel).stem))

    seen_sources: set[str] = set()
    while queue:
        path, source, text_stem = queue.pop(0)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        if not path.is_file():
            meta.notes.append(f"missing file: {source}")
            continue
        try:
            text, method = extract_file(path)
        except Exception as exc:  # noqa: BLE001
            extracted.append(
                {
                    "source": source,
                    "method": "error",
                    "error": str(exc),
                }
            )
            meta.notes.append(f"extract failed: {source}: {exc}")
            continue

        if method.startswith("archive:"):
            unpack_rel = f"files/{Path(source).stem}_unpacked"
            unpack_dir = (out_dir / unpack_rel).resolve()
            try:
                members = safe_unpack_zip(path, unpack_dir)
            except Exception as exc:  # noqa: BLE001
                extracted.append(
                    {"source": source, "method": "error", "error": str(exc)}
                )
                meta.notes.append(f"unpack failed: {source}: {exc}")
                continue
            extracted.append(
                {
                    "source": source,
                    "method": method,
                    "unpacked_to": unpack_rel,
                    "members": len(members),
                }
            )
            for member in members:
                member = member.resolve()
                try:
                    rel_member = str(member.relative_to(out_dir.resolve()))
                except ValueError:
                    rel_member = str(member)
                inner = member.relative_to(unpack_dir)
                queue.append(
                    (
                        member,
                        rel_member,
                        f"{Path(source).stem}/{inner.as_posix()}",
                    )
                )
            continue

        if method.startswith("unsupported"):
            extracted.append({"source": source, "method": method, "chars": 0})
            meta.notes.append(f"unsupported format: {source} ({method})")
            continue

        out_path = text_dir / Path(text_stem).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        text_rel = out_path.relative_to(text_dir).as_posix()
        extracted.append(
            {
                "source": source,
                "text_path": f"text/{text_rel}",
                "method": method,
                "chars": len(text),
            }
        )
        if text:
            combined_parts.append(f"\n\n===== {source} =====\n\n{text}")

    combined = normalize_text("\n\n".join(combined_parts))
    (text_dir / "_combined.txt").write_text(combined + "\n", encoding="utf-8")
    meta.extracted = extracted
    return meta


def write_meta(meta: ComplaintMeta, out_dir: Path) -> None:
    (out_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fetch_complaint(
    complaint_number: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 90.0,
    skip_download: bool = False,
    insecure: bool = False,
) -> Path:
    out_dir = complaint_dir(out_root, complaint_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = card_urls(complaint_number)
    # Сначала актуальная вкладка; устаревшие common/documents — только fallback.
    preferred = ("information", "common", "documents")
    meta: ComplaintMeta | None = None
    got_html = False
    fallback_notes: list[str] = []

    for key in preferred:
        url = urls[key]
        try:
            data, _ctype, final = http_get(
                url, timeout=timeout, sleep_s=sleep_s, insecure=insecure
            )
            time.sleep(sleep_s)
            html = decode_html(data)
            if len(html.strip()) < 200:
                raise RuntimeError(
                    f"слишком короткий ответ ({len(html)} символов) — "
                    "возможно, блокировка/капча или неверный номер жалобы"
                )
            piece = ingest_card_html(
                complaint_number=complaint_number,
                html=html,
                out_dir=out_dir,
                card_name=f"card_{key}.html",
                card_url=final,
            )
            got_html = True
            meta = piece
            break  # одной рабочей вкладки достаточно
        except Exception as exc:  # noqa: BLE001
            note = f"card {key} failed: {exc}"
            # 404 по устаревшим вкладкам не шумим в stderr, если information уже ок
            # (до break сюда не дойдём). Пока ищем — пишем только не-404 или первый ключ.
            if key == "information" or "404" not in str(exc):
                print(note, file=sys.stderr)
            fallback_notes.append(note)
            if meta is None:
                meta = ComplaintMeta(
                    complaint_number=complaint_number,
                    card_url=url,
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                    notes=list(fallback_notes),
                )
            else:
                meta.notes.append(note)

    assert meta is not None
    # Не оставляем шум от устаревших 404, если information уже скачана.
    if got_html:
        meta.notes = [
            n
            for n in meta.notes
            if not (
                n.startswith("card common failed:")
                or n.startswith("card documents failed:")
            )
        ]
    if not got_html:
        write_meta(meta, out_dir)
        hint = ""
        if any("CERTIFICATE_VERIFY_FAILED" in n for n in meta.notes):
            hint = (
                "\nПохоже на SSL/прокси. Повторите с --insecure или сохраните карточку "
                "из браузера и: fetch --from-html card.html --complaint-number …"
            )
        raise RuntimeError(
            f"Не удалось скачать HTML карточки жалобы {complaint_number}. "
            f"Смотрите {out_dir / 'meta.json'}.{hint}"
        )

    if not skip_download and meta.documents:
        meta = download_listed_documents(
            meta, out_dir, sleep_s=sleep_s, timeout=timeout, insecure=insecure
        )
    meta = extract_all_files(meta, out_dir)
    write_meta(meta, out_dir)
    return out_dir


def fetch_from_html_file(
    html_path: Path,
    complaint_number: str,
    out_root: Path,
    *,
    download: bool = False,
    sleep_s: float = 1.5,
    insecure: bool = False,
) -> Path:
    html = html_path.read_text(encoding="utf-8")
    out_dir = complaint_dir(out_root, complaint_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = ingest_card_html(
        complaint_number=complaint_number,
        html=html,
        out_dir=out_dir,
        card_name="card_common.html",
        card_url=None,
    )
    if download and meta.documents:
        meta = download_listed_documents(
            meta, out_dir, sleep_s=sleep_s, insecure=insecure
        )
    meta = extract_all_files(meta, out_dir)
    write_meta(meta, out_dir)
    return out_dir


def _collect_numbers(args: argparse.Namespace) -> list[str]:
    numbers: list[str] = []
    for attr in ("number", "complaint_number"):
        vals = getattr(args, attr, None) or []
        numbers.extend(vals)
    if getattr(args, "numbers_file", None):
        for line in Path(args.numbers_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                numbers.append(line)
    return numbers


def cmd_complaint(args: argparse.Namespace) -> int:
    """Скачать жалобу по номеру (или из сохранённого HTML карточки)."""
    out_root = Path(args.out_root)
    numbers = _collect_numbers(args)
    if args.url and not numbers:
        m = COMPLAINT_NUM_RE.search(args.url)
        if not m:
            print("В --url нет complaintNumber=…", file=sys.stderr)
            return 2
        numbers.append(m.group(1))

    if args.from_html:
        if not numbers:
            print("Для --from-html укажите --number <номер жалобы>", file=sys.stderr)
            return 2
        path = fetch_from_html_file(
            Path(args.from_html),
            numbers[0],
            out_root,
            download=not args.skip_download,
            sleep_s=args.sleep,
            insecure=args.insecure,
        )
        print(path)
        if args.with_notice:
            attach_notices(
                path,
                Path(args.notice_out_root),
                fetch_missing=not args.link_only,
                sleep_s=args.sleep,
                timeout=args.timeout,
                insecure=args.insecure,
            )
        return 0

    if not numbers:
        print(
            "Укажите --number <номер жалобы ЕИС> (не номер извещения).\n"
            "Извещение: python scripts/download_fas.py notice --number <regNumber>\n"
            "Связать уже скачанное: python scripts/download_fas.py link "
            "--complaint-dir data/raw/fas/<номер>",
            file=sys.stderr,
        )
        return 2

    failed = 0
    for num in numbers:
        print(f"Fetching complaint {num} …", file=sys.stderr)
        try:
            path = fetch_complaint(
                num,
                out_root,
                sleep_s=args.sleep,
                timeout=args.timeout,
                skip_download=args.skip_download,
                insecure=args.insecure,
            )
            print(path)
            if args.with_notice:
                linked = attach_notices(
                    path,
                    Path(args.notice_out_root),
                    fetch_missing=not args.link_only,
                    sleep_s=args.sleep,
                    timeout=args.timeout,
                    insecure=args.insecure,
                )
                for npath in linked:
                    print(npath)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            failed += 1
        time.sleep(args.sleep)
    return 1 if failed else 0


def cmd_link(args: argparse.Namespace) -> int:
    """Связать уже скачанную жалобу с извещением(ями) без повторной загрузки жалобы."""
    complaint_dir = Path(args.complaint_dir)
    if args.notice_number:
        # временно дописать номер в meta, если автообнаружение не сработает
        meta_path = complaint_dir / "meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        pids = normalize_procurement_ids(
            list(raw.get("procurement_ids") or []) + list(args.notice_number),
            exclude={raw.get("complaint_number") or complaint_dir.name},
        )
        raw["procurement_ids"] = pids
        meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        linked = attach_notices(
            complaint_dir,
            Path(args.notice_out_root),
            fetch_missing=not args.link_only,
            sleep_s=args.sleep,
            timeout=args.timeout,
            insecure=args.insecure,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for p in linked:
        print(p)
    return 0 if linked else 1


def cmd_notice(args: argparse.Namespace) -> int:
    from ingest.notices.download import fetch_notice

    out_root = Path(args.out_root)
    numbers = _collect_numbers(args)
    if not numbers:
        print("Укажите --number <regNumber извещения>", file=sys.stderr)
        return 2

    failed = 0
    for num in numbers:
        print(f"Fetching notice {num} …", file=sys.stderr)
        try:
            path = fetch_notice(
                num,
                out_root,
                sleep_s=args.sleep,
                timeout=args.timeout,
                skip_download=args.skip_download,
                insecure=args.insecure,
                kind=args.kind,
            )
            print(path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            failed += 1
        time.sleep(args.sleep)
    return 1 if failed else 0


def cmd_extract(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = ComplaintMeta(
            complaint_number=raw.get("complaint_number")
            or raw.get("procurement_id")
            or out_dir.name,
            documents=raw.get("documents") or [],
            extracted=raw.get("extracted") or [],
            notes=raw.get("notes") or [],
            downloaded_at=raw.get("downloaded_at"),
            procurement_ids=raw.get("procurement_ids") or [],
            notice_dirs=raw.get("notice_dirs") or [],
        )
    else:
        meta = ComplaintMeta(complaint_number=out_dir.name)
        files_dir = out_dir / "files"
        if files_dir.exists():
            meta.documents = [
                {"url": "", "title": p.name, "status": "downloaded", "path": f"files/{p.name}"}
                for p in sorted(files_dir.iterdir())
                if p.is_file()
            ]
    meta = extract_all_files(meta, out_dir)
    write_meta(meta, out_dir)
    print(out_dir / "text" / "_combined.txt")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sleep", type=float, default=1.5, help="Пауза между запросами, сек")
    common.add_argument("--timeout", type=float, default=90.0)
    common.add_argument(
        "--insecure",
        action="store_true",
        help="Не проверять SSL (корпоративный прокси / self-signed в цепочке)",
    )

    p = argparse.ArgumentParser(
        description="Скачать жалобу или извещение ЕИС по номеру + сырой текст (без разметки EvalCase)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("complaint", parents=[common], help="Жалоба по номеру complaintNumber")
    c.add_argument("--out-root", type=Path, default=Path("data/raw/fas"))
    c.add_argument(
        "--number",
        "--complaint-number",
        dest="number",
        action="append",
        default=[],
        help="Номер жалобы ЕИС",
    )
    c.add_argument("--numbers-file", help="Файл с номерами жалоб")
    c.add_argument("--url", help="URL карточки с complaintNumber=")
    c.add_argument("--from-html", help="Локальный HTML карточки (офлайн)")
    c.add_argument("--skip-download", action="store_true", help="Только HTML, без вложений")
    c.add_argument(
        "--with-notice",
        action="store_true",
        help="Связать с извещением: reuse data/raw/notices/{id}, иначе докачать",
    )
    c.add_argument(
        "--link-only",
        action="store_true",
        help="С --with-notice: только связать уже скачанные извещения, без fetch",
    )
    c.add_argument(
        "--notice-out-root",
        type=Path,
        default=Path("data/raw/notices"),
        help="Куда класть/искать извещения при --with-notice",
    )
    c.set_defaults(func=cmd_complaint, link_only=False)

    l = sub.add_parser(
        "link",
        parents=[common],
        help="Связать уже скачанную жалобу с извещением (без повторной загрузки жалобы)",
    )
    l.add_argument(
        "--complaint-dir",
        type=Path,
        required=True,
        help="data/raw/fas/{номер жалобы}",
    )
    l.add_argument(
        "--notice-number",
        action="append",
        default=[],
        help="Номер извещения, если не находится в meta/тексте жалобы",
    )
    l.add_argument(
        "--link-only",
        action="store_true",
        help="Не докачивать отсутствующие извещения",
    )
    l.add_argument(
        "--notice-out-root",
        type=Path,
        default=Path("data/raw/notices"),
    )
    l.set_defaults(func=cmd_link)

    n = sub.add_parser("notice", parents=[common], help="Извещение по regNumber закупки")
    n.add_argument("--out-root", type=Path, default=Path("data/raw/notices"))
    n.add_argument("--number", action="append", default=[], help="Номер извещения (regNumber)")
    n.add_argument("--numbers-file", help="Файл с номерами извещений")
    n.add_argument(
        "--kind",
        choices=["ea20", "ea44", "ok20", "ok44", "zk20", "zk44", "ep44", "ezt20", "ezt44"],
        help="Тип извещения в URL (если известен; иначе автоподбор)",
    )
    n.add_argument("--skip-download", action="store_true", help="Только HTML, без вложений")
    n.set_defaults(func=cmd_notice)

    e = sub.add_parser("extract", parents=[common], help="Переизвлечь текст из уже скачанной папки")
    e.add_argument("--out-root", type=Path, default=Path("data/raw/fas"), help=argparse.SUPPRESS)
    e.add_argument("--dir", required=True, help="data/raw/fas/{номер} или data/raw/notices/{номер}")
    e.set_defaults(func=cmd_extract)

    # совместимость со старым именем подкоманды
    f = sub.add_parser("fetch", parents=[common], help=argparse.SUPPRESS)
    f.add_argument("--out-root", type=Path, default=Path("data/raw/fas"))
    f.add_argument("--number", "--complaint-number", dest="number", action="append", default=[])
    f.add_argument("--numbers-file")
    f.add_argument("--url")
    f.add_argument("--from-html")
    f.add_argument("--skip-download", action="store_true")
    f.add_argument("--with-notice", action="store_true")
    f.add_argument("--link-only", action="store_true")
    f.add_argument("--notice-out-root", type=Path, default=Path("data/raw/notices"))
    f.set_defaults(func=cmd_complaint, link_only=False)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
