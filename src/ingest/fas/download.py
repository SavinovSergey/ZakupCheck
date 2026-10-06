"""Скачивание жалоб ФАС / карточек ЕИС + сырой текст (DESIGN §4.5).

Не размечает EvalCase — только кэш файлов и извлечённый текст для ручной разметки.

  data/raw/fas/{complaint_number}/     — жалоба
  data/raw/notices/{procurement_id}/   — извещение (подкоманда notice)

Поиск по произвольному query убран: номера жалобы/извещения задаются явно.
"""

from __future__ import annotations

import argparse
import asyncio
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
    heavy_kind,
    normalize_text,
    safe_unpack_zip,
)

USER_AGENT = "ZakupCheck/0.1 (+local research; respectful crawl; FAS eval ingest)"
EIS_BASE = "https://zakupki.gov.ru"
# После успешного ответа — короткая пауза. Длинная пауза (sleep_s) — между жалобами и после 403.
SUCCESS_PAUSE_S = 0.1
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
# После 403 не переключаемся на соседние URL: ждём и повторяем тот же запрос.
FORBIDDEN_COOLDOWN_S = (20, 40)

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
LONG_ID_RE = re.compile(r"\b(\d{18,19})\b")
# поле «Номер извещения» в карточке жалобы ЕИС (основной источник)
NOTICE_LABEL_RE = re.compile(
    r"Номер\s+извещения\s*[№N#:]?\s*(\d{18,19})",
    re.I,
)
# контекст в произвольном тексте: извещение / закупка vs реестр контракта
NOTICE_CTX_RE = re.compile(
    r"(?:номер\w*\s+извещен|извещен\w*\s*[№N#]|извещен\w*\s+об\s|"
    r"номер\w*\s+закупк|закупк\w*\s*[№N#]|аукцион\w*\s*[№N#]|"
    r"regNumber|reestrNumber|purchaseNumber|/epz/order/notice/)",
    re.I,
)
CONTRACT_CTX_RE = re.compile(
    r"(?:реестров\w*\s+номер\w*\s+контракт|"
    r"контракт\w*\s+с\s+реестров\w*\s+номер|"
    r"государственн\w*\s+контракт\w*[^.…]{0,40}реестров|"
    r"номер\w*\s+контракт\w*\s+в\s+реестр|"
    r"реестр\w*\s+контракт)",
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


class DownloadRejected(Exception):
    """Вложение не сохраняем: .bin или больше лимита. Тело по возможности не читаем."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def would_be_bin(url: str, title: str | None, content_type: str) -> bool:
    """Имя, под которым файл лёг бы на диск, — *.bin."""
    return Path(guess_filename(url, title, content_type, 1)).suffix.lower() == ".bin"


def _read_limited(resp: object, max_bytes: int | None) -> bytes:
    read = getattr(resp, "read")
    if max_bytes is None:
        return read()
    chunks: list[bytes] = []
    total = 0
    while True:
        block = read(64 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise DownloadRejected("too-large")
        chunks.append(block)
    return b"".join(chunks)


def http_get(
    url: str,
    *,
    timeout: float = 15.0,
    retries: int = 3,
    sleep_s: float = 1.0,
    insecure: bool = False,
    max_bytes: int | None = None,
    title: str | None = None,
    reject_bin: bool = False,
) -> tuple[bytes, str, str]:
    """Возвращает (body, content_type, final_url).

    После успеха пауза SUCCESS_PAUSE_S. sleep_s — пауза повтора после HTTP 403.
    """
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
                ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                final = resp.geturl()
                if reject_bin and would_be_bin(final or url, title, ctype):
                    raise DownloadRejected("bin")
                length = resp.headers.get("Content-Length")
                if max_bytes is not None and length:
                    try:
                        if int(length) > max_bytes:
                            raise DownloadRejected("too-large")
                    except ValueError:
                        pass
                data = _read_limited(resp, max_bytes)
            time.sleep(SUCCESS_PAUSE_S)
            return data, ctype, final
        except DownloadRejected:
            raise
        except urllib.error.HTTPError as exc:
            last_err = exc
            retriable = exc.code in {403, 408, 425, 429} or 500 <= exc.code < 600
            if not retriable:
                raise
            if attempt >= retries:
                break
            pause = sleep_s if exc.code == 403 else min(2**attempt, 4)
            time.sleep(pause)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 4))
    assert last_err is not None
    raise last_err


async def download_url(url: str, **kwargs: object) -> tuple[bytes, str, str]:
    """HTTP-запрос в потоке, чтобы цикл событий мог в это время вести OCR."""
    return await asyncio.to_thread(http_get, url, **kwargs)


async def extract_heavy(path: Path) -> tuple[str, str]:
    """OCR или LibreOffice. Не занимает цикл, в котором идут скачивания."""
    return await asyncio.to_thread(extract_file, path)


def is_explicit_bin(url: str, title: str | None) -> bool:
    """Ссылка или заголовок уже называются *.bin — запрос не делаем."""
    path = urllib.parse.urlparse(url).path
    if Path(path).name.lower().endswith(".bin"):
        return True
    if title and Path(title.strip()).suffix.lower() == ".bin":
        return True
    return False


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit_download(dest: Path, data: bytes) -> tuple[Path, str]:
    """Запись файла, проверка типа и хеш — вне цикла событий."""
    save_bytes(dest, data)
    return correct_suffix(dest), sha256_bytes(data)


async def _persist_heavy(out_dir: Path, path: Path, source: str, text_stem: str) -> dict:
    """Записать текст тяжёлого файла. meta.json здесь не трогаем."""
    try:
        text, method = await extract_heavy(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "out_dir": str(out_dir),
            "source": source,
            "method": "error",
            "error": str(exc),
        }
    text_dir = out_dir / "text"
    out_path = text_dir / Path(text_stem).with_suffix(".txt")
    await asyncio.to_thread(_write_text, out_path, text + "\n")
    text_rel = out_path.relative_to(text_dir).as_posix()
    return {
        "out_dir": str(out_dir),
        "source": source,
        "text_path": f"text/{text_rel}",
        "method": method,
        "chars": len(text),
    }


def _rebuild_combined(out_dir: Path, extracted: list[dict]) -> None:
    parts: list[str] = []
    for entry in extracted:
        if entry.get("method") != "html" or not entry.get("text_path"):
            continue
        path = out_dir / entry["text_path"]
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    for entry in extracted:
        method = str(entry.get("method") or "")
        if method in {"html", "pending", "error"} or method.startswith(("archive:", "unsupported")):
            continue
        if not entry.get("text_path"):
            continue
        path = out_dir / entry["text_path"]
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").rstrip("\n")
        parts.append(f"\n\n===== {entry.get('source')} =====\n\n{text}")
    combined = normalize_text("\n\n".join(parts))
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "_combined.txt").write_text(combined + "\n", encoding="utf-8")


def _apply_heavy(out_dir: Path, items: list[dict]) -> None:
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        return
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    by_source = {item["source"]: item for item in items}
    replaced: set[str] = set()
    extracted: list[dict] = []
    notes = list(raw.get("notes") or [])
    for entry in raw.get("extracted") or []:
        source = entry.get("source")
        item = by_source.get(source) if entry.get("method") == "pending" else None
        if not item:
            extracted.append(entry)
            continue
        replaced.add(source)
        if item.get("method") == "error":
            extracted.append({"source": source, "method": "error", "error": item.get("error")})
            notes.append(f"extract failed: {source}: {item.get('error')}")
        else:
            extracted.append(
                {
                    "source": source,
                    "text_path": item["text_path"],
                    "method": item["method"],
                    "chars": item["chars"],
                }
            )
    for source, item in by_source.items():
        if source in replaced:
            continue
        if item.get("method") == "error":
            extracted.append({"source": source, "method": "error", "error": item.get("error")})
            notes.append(f"extract failed: {source}: {item.get('error')}")
        else:
            extracted.append(
                {
                    "source": source,
                    "text_path": item["text_path"],
                    "method": item["method"],
                    "chars": item["chars"],
                }
            )
    for doc in raw.get("documents") or []:
        doc.pop("deferred", None)
    raw["extracted"] = extracted
    raw["notes"] = notes
    _rebuild_combined(out_dir, extracted)
    meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def finish_heavy(tasks: list) -> list[dict]:
    """Дождаться OCR/LibreOffice и дописать meta.json.

    Возвращает ошибки служебной обработки; ошибки извлечения
    отдельных файлов сами попадают в meta.json как method=error.
    """
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    grouped: dict[str, list[dict]] = {}
    errors: list[dict] = []
    for item in results:
        if isinstance(item, Exception):
            print(f"heavy extract failed: {item}", file=sys.stderr)
            errors.append({"out_dir": None, "error": str(item)})
            continue
        if not item:
            continue
        grouped.setdefault(item["out_dir"], []).append(item)
    for dir_s, items in grouped.items():
        try:
            _apply_heavy(Path(dir_s), items)
        except Exception as exc:  # noqa: BLE001 — остальные meta всё равно обновляем
            print(f"heavy meta update failed for {dir_s}: {exc}", file=sys.stderr)
            errors.append({"out_dir": dir_s, "error": str(exc)})
    tasks.clear()
    return errors


def _is_forbidden(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == 403 or "403" in str(exc)


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


def _nearest_ctx_dist(window: str, id_offset: int, pattern: re.Pattern[str]) -> int | None:
    """Минимальное расстояние от id до совпадения pattern внутри window; None если нет."""
    best: int | None = None
    for cm in pattern.finditer(window):
        # расстояние от ближайшего края совпадения до позиции id
        if cm.end() <= id_offset:
            d = id_offset - cm.end()
        elif cm.start() >= id_offset:
            d = cm.start() - id_offset
        else:
            d = 0
        if best is None or d < best:
            best = d
    return best


def classify_long_ids(text: str) -> tuple[set[str], set[str], set[str]]:
    """Разнести 18–19-значные id по ближайшему контексту: notice / contract / unknown."""
    notice: set[str] = set()
    contract: set[str] = set()
    unknown: set[str] = set()
    # узкое окно: дальше 60 символов контекст обычно уже про другое
    radius = 60
    for m in LONG_ID_RE.finditer(text):
        num = m.group(1)
        left = max(0, m.start() - radius)
        window = text[left : m.end() + radius]
        id_offset = m.start() - left
        d_notice = _nearest_ctx_dist(window, id_offset, NOTICE_CTX_RE)
        d_contract = _nearest_ctx_dist(window, id_offset, CONTRACT_CTX_RE)
        if d_notice is None and d_contract is None:
            unknown.add(num)
        elif d_contract is None or (d_notice is not None and d_notice < d_contract):
            notice.add(num)
        elif d_notice is None or d_contract < d_notice:
            contract.add(num)
        else:
            # одинаково близко — безопаснее считать контрактом (не тащить в probe)
            contract.add(num)
    return notice, contract, unknown


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


def _is_card_text_name(name: str) -> bool:
    low = name.lower()
    return low.startswith("card_") or "сведен" in low or "complaint" in low


def _is_card_html_path(path: Path) -> bool:
    """HTML карточки жалобы (не произвольные вложения с чужими ссылками)."""
    if path.parent.name == "raw" and path.suffix.lower() == ".html":
        return True
    low = path.name.lower()
    return path.suffix.lower() == ".html" and (
        "сведен" in low or "complaint" in low or low.startswith("card_")
    )


def parse_notice_labels(text: str) -> list[str]:
    return NOTICE_LABEL_RE.findall(text)


def discover_procurement_ids_from_complaint_dir(
    out_dir: Path,
    *,
    complaint_number: str,
) -> list[str]:
    """Номер извещения для жалобы: сначала карточка ЕИС, иначе осторожный fallback.

    На 50 скачанных жалобах поле «Номер извещения» / HTML карточки всегда даёт
    нужный id; дополнительные номера из текста жалобы/решения — чужие примеры
    (ложный массовый fetch). Имена файлов не используем.
    """
    exclude = {complaint_number}
    card_ids: list[str] = []

    for p in out_dir.glob("raw/*.html"):
        if not _is_card_html_path(p):
            continue
        try:
            html = p.read_text(encoding="utf-8")
        except OSError:
            continue
        card_ids.extend(ORDER_NOTICE_RE.findall(html))
        card_ids.extend(REG_NUMBER_RE.findall(html))
        card_ids.extend(parse_notice_labels(html))

    for p in out_dir.glob("files/*.html"):
        if not _is_card_html_path(p):
            continue
        try:
            html = p.read_text(encoding="utf-8")
        except OSError:
            continue
        card_ids.extend(ORDER_NOTICE_RE.findall(html))
        card_ids.extend(REG_NUMBER_RE.findall(html))
        card_ids.extend(parse_notice_labels(html))

    for p in out_dir.glob("text/*"):
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        if not _is_card_text_name(p.name):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        card_ids.extend(parse_notice_labels(text))

    card_norm = normalize_procurement_ids(card_ids, exclude=exclude)
    if card_norm:
        return card_norm

    # fallback: meta (уже связанные) + текст только с контекстом «извещение»
    fallback: list[str] = []
    contract_ids: set[str] = set()
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        # не доверяем раздутому procurement_ids из прошлых прогонов — только notice_dirs
        fallback.extend(
            Path(p).name
            for p in (raw.get("notice_dirs") or [])
            if Path(p).name and Path(p).name.isdigit()
        )

    for p in out_dir.glob("text/*"):
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        if p.name.startswith("_") or _is_card_text_name(p.name):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        n_ids, c_ids, _unk = classify_long_ids(text)
        fallback.extend(n_ids)
        contract_ids.update(c_ids)

    out = [
        x
        for x in normalize_procurement_ids(fallback, exclude=exclude)
        if x not in contract_ids
    ]
    if contract_ids:
        dropped = sorted(contract_ids - set(out))
        if dropped:
            print(
                f"skip contract-like ids (not notice): {', '.join(dropped[:5])}"
                + ("…" if len(dropped) > 5 else ""),
                file=sys.stderr,
            )
    if len(out) > 1:
        print(
            f"fallback notice ids ({len(out)}): using all candidates; "
            f"prefer card «Номер извещения» when available",
            file=sys.stderr,
        )
    return out


async def attach_notices_async(
    complaint_dir: Path,
    notice_out_root: Path,
    *,
    fetch_missing: bool = True,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    insecure: bool = False,
    heavy_tasks: list | None = None,
) -> list[Path]:
    """Связать жалобу с извещениями: использовать уже скачанные или докачать.

    Пишет в meta жалобы: procurement_ids (очищенные) и notice_dirs.
    heavy_tasks — общий список OCR/LibreOffice, если этапы стыкует вызывающий код.
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

    from ingest.notices.download import fetch_notice_async

    own_tasks = heavy_tasks is None
    tasks: list = [] if own_tasks else heavy_tasks
    try:
        for pid in pids:
            npath = notice_out_root / pid
            if (npath / "meta.json").exists():
                print(f"Link existing notice {pid} → {npath}", file=sys.stderr)
            elif fetch_missing:
                print(f"Fetching linked notice {pid} …", file=sys.stderr)
                npath = await fetch_notice_async(
                    pid,
                    notice_out_root,
                    sleep_s=sleep_s,
                    timeout=timeout,
                    insecure=insecure,
                    heavy_tasks=tasks,
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
    finally:
        if own_tasks:
            await finish_heavy(tasks)


def attach_notices(
    complaint_dir: Path,
    notice_out_root: Path,
    *,
    fetch_missing: bool = True,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    insecure: bool = False,
) -> list[Path]:
    """Синхронная обёртка: дожидается и скачивания, и OCR/LibreOffice."""
    return asyncio.run(
        attach_notices_async(
            complaint_dir,
            notice_out_root,
            fetch_missing=fetch_missing,
            sleep_s=sleep_s,
            timeout=timeout,
            insecure=insecure,
        )
    )


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


async def download_listed_documents(
    meta: ComplaintMeta,
    out_dir: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 120.0,
    insecure: bool = False,
    heavy_tasks: list | None = None,
) -> ComplaintMeta:
    """Скачать вложения по одному. Тяжёлый разбор ставится в heavy_tasks и не ждётся."""
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict] = []
    for i, doc in enumerate(meta.documents, start=1):
        url = doc["url"]
        title = doc.get("title")
        if is_explicit_bin(url, title):
            updated.append({**doc, "status": "skipped", "reason": "bin"})
            meta.notes.append(f"skip bin: {title or url}")
            continue
        try:
            data, ctype, final_url = await download_url(
                url,
                timeout=timeout,
                sleep_s=sleep_s,
                insecure=insecure,
                max_bytes=MAX_ATTACHMENT_BYTES,
                title=title,
                reject_bin=True,
            )
            fname = guess_filename(final_url or url, title, ctype, i)
            dest = files_dir / fname
            if dest.exists():
                dest = files_dir / f"{dest.stem}_{i}{dest.suffix}"
            fixed, digest = await asyncio.to_thread(_commit_download, dest, data)
            if fixed != dest:
                meta.notes.append(f"renamed by magic: {dest.name} → {fixed.name}")
                dest = fixed
            if fixed.suffix.lower() == ".bin":
                fixed.unlink(missing_ok=True)
                updated.append({**doc, "status": "skipped", "reason": "bin"})
                meta.notes.append(f"skip bin after sniff: {title or url}")
                continue
            record = {
                **doc,
                "status": "downloaded",
                "path": f"files/{fixed.name}",
                "content_type": ctype,
                "sha256": digest,
                "bytes": len(data),
                "final_url": final_url,
            }
            if heavy_tasks is not None:
                try:
                    kind = await asyncio.to_thread(heavy_kind, fixed)
                except Exception as exc:  # noqa: BLE001
                    meta.notes.append(f"heavy check failed: {fixed.name}: {exc}")
                    kind = None
                if kind:
                    record["deferred"] = kind
                    heavy_tasks.append(
                        asyncio.create_task(
                            _persist_heavy(out_dir, fixed, record["path"], Path(record["path"]).stem)
                        )
                    )
            updated.append(record)
        except DownloadRejected as exc:
            updated.append({**doc, "status": "skipped", "reason": exc.reason})
            meta.notes.append(f"skip {exc.reason}: {title or url}")
        except Exception as exc:  # noqa: BLE001 — копим ошибки по файлам
            updated.append({**doc, "status": "error", "error": str(exc)})
            meta.notes.append(f"download failed: {url}: {exc}")
    meta.documents = updated
    return meta


async def extract_all_files_async(
    meta: ComplaintMeta,
    out_dir: Path,
    heavy_tasks: list | None = None,
) -> ComplaintMeta:
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
        if doc.get("deferred"):
            extracted.append(
                {"source": rel, "method": "pending", "heavy": doc["deferred"]}
            )
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
        if heavy_tasks is not None:
            try:
                kind = await asyncio.to_thread(heavy_kind, path)
            except Exception as exc:  # noqa: BLE001
                meta.notes.append(f"heavy check failed: {source}: {exc}")
                kind = None
            if kind:
                heavy_tasks.append(
                    asyncio.create_task(_persist_heavy(out_dir, path, source, text_stem))
                )
                extracted.append({"source": source, "method": "pending", "heavy": kind})
                continue
        try:
            text, method = await asyncio.to_thread(extract_file, path)
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
                members = await asyncio.to_thread(safe_unpack_zip, path, unpack_dir)
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
        await asyncio.to_thread(_write_text, out_path, text + "\n")
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
    await asyncio.to_thread(_write_text, text_dir / "_combined.txt", combined + "\n")
    meta.extracted = extracted
    return meta


def extract_all_files(
    meta: ComplaintMeta,
    out_dir: Path,
    heavy_tasks: list | None = None,
) -> ComplaintMeta:
    """Синхронный разбор, когда цикла событий нет (офлайн-карточка, команда extract)."""
    return asyncio.run(extract_all_files_async(meta, out_dir, heavy_tasks))


def write_meta(meta: ComplaintMeta, out_dir: Path) -> None:
    (out_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def fetch_complaint_async(
    complaint_number: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    skip_download: bool = False,
    insecure: bool = False,
    heavy_tasks: list | None = None,
) -> Path:
    out_dir = complaint_dir(out_root, complaint_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = card_urls(complaint_number)
    # Сначала актуальная вкладка; устаревшие common/documents — только fallback.
    preferred = ("information", "common", "documents")
    own_tasks = heavy_tasks is None
    tasks: list = [] if own_tasks else heavy_tasks
    try:
        return await _fetch_complaint_body(
            complaint_number,
            out_dir,
            urls,
            preferred,
            tasks,
            sleep_s=sleep_s,
            timeout=timeout,
            skip_download=skip_download,
            insecure=insecure,
        )
    finally:
        if own_tasks:
            await finish_heavy(tasks)


async def _fetch_complaint_body(
    complaint_number: str,
    out_dir: Path,
    urls: dict[str, str],
    preferred: tuple[str, ...],
    tasks: list,
    *,
    sleep_s: float,
    timeout: float,
    skip_download: bool,
    insecure: bool,
) -> Path:
    meta: ComplaintMeta | None = None
    got_html = False
    fallback_notes: list[str] = []
    forbidden = False

    async def _try_card(key: str) -> ComplaintMeta:
        url = urls[key]
        data, _ctype, final = await download_url(
            url, timeout=timeout, sleep_s=sleep_s, insecure=insecure
        )
        html = await asyncio.to_thread(decode_html, data)
        if len(html.strip()) < 200:
            raise RuntimeError(
                f"слишком короткий ответ ({len(html)} символов) — "
                "возможно, блокировка/капча или неверный номер жалобы"
            )
        return await asyncio.to_thread(
            ingest_card_html,
            complaint_number=complaint_number,
            html=html,
            out_dir=out_dir,
            card_name=f"card_{key}.html",
            card_url=final,
        )

    for key in preferred:
        try:
            meta = await _try_card(key)
            got_html = True
            break  # одной рабочей вкладки достаточно
        except Exception as exc:  # noqa: BLE001
            note = f"card {key} failed: {exc}"
            # 404 по устаревшим вкладкам не шумим в stderr, если information уже ок
            # (до break сюда не дойдём). Пока ищем — пишем только не-404 или первый ключ.
            if key == "information" or "404" not in str(exc):
                print(note, file=sys.stderr)
            fallback_notes.append(note)
            if _is_forbidden(exc):
                # 403 — ограничение частоты, а не отсутствие вкладки.
                forbidden = True
                break
            if meta is None:
                meta = ComplaintMeta(
                    complaint_number=complaint_number,
                    card_url=urls[key],
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                    notes=list(fallback_notes),
                )
            else:
                meta.notes.append(note)

    if not got_html and forbidden:
        for pause in FORBIDDEN_COOLDOWN_S:
            print(
                f"403 у карточки {complaint_number}: пауза {pause:.0f} с и повтор",
                file=sys.stderr,
            )
            await asyncio.sleep(pause)
            try:
                meta = await _try_card("information")
                got_html = True
                break
            except Exception as exc:  # noqa: BLE001
                note = f"card information failed: {exc}"
                print(note, file=sys.stderr)
                fallback_notes.append(note)
                if not _is_forbidden(exc):
                    break

    if got_html:
        assert meta is not None
        # Не оставляем шум от устаревших 404, если information уже скачана.
        meta.notes = [
            n
            for n in meta.notes
            if not (
                n.startswith("card common failed:")
                or n.startswith("card documents failed:")
            )
        ]
    if not got_html:
        # 403 не записываем как готовую карточку: иначе пакет больше её не возьмёт.
        notes = meta.notes if meta is not None else fallback_notes
        if not forbidden and meta is not None:
            write_meta(meta, out_dir)
        hint = ""
        if any("CERTIFICATE_VERIFY_FAILED" in n for n in notes):
            hint = (
                "\nПохоже на SSL/прокси. Повторите с --insecure или сохраните карточку "
                "из браузера и: fetch --from-html card.html --complaint-number …"
            )
        where = (
            "Повтор при следующем запуске, meta.json не записан."
            if forbidden
            else f"Смотрите {out_dir / 'meta.json'}."
        )
        detail = fallback_notes[-1] if fallback_notes else "причина неизвестна"
        raise RuntimeError(
            f"Не удалось скачать HTML карточки жалобы {complaint_number}. "
            f"{where} Последняя ошибка: {detail}{hint}"
        )

    if not skip_download and meta.documents:
        meta = await download_listed_documents(
            meta,
            out_dir,
            sleep_s=sleep_s,
            timeout=timeout,
            insecure=insecure,
            heavy_tasks=tasks,
        )
    meta = await extract_all_files_async(meta, out_dir, heavy_tasks=tasks)
    write_meta(meta, out_dir)
    return out_dir


def fetch_complaint(
    complaint_number: str,
    out_root: Path,
    *,
    sleep_s: float = 1.5,
    timeout: float = 15.0,
    skip_download: bool = False,
    insecure: bool = False,
) -> Path:
    """Синхронная обёртка: дожидается и скачивания, и OCR/LibreOffice."""
    return asyncio.run(
        fetch_complaint_async(
            complaint_number,
            out_root,
            sleep_s=sleep_s,
            timeout=timeout,
            skip_download=skip_download,
            insecure=insecure,
        )
    )


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
        meta = asyncio.run(
            download_listed_documents(meta, out_dir, sleep_s=sleep_s, insecure=insecure)
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

    async def _run() -> int:
        failed = 0
        tasks: list = []
        try:
            for num in numbers:
                print(f"Fetching complaint {num} …", file=sys.stderr)
                try:
                    path = await fetch_complaint_async(
                        num,
                        out_root,
                        sleep_s=args.sleep,
                        timeout=args.timeout,
                        skip_download=args.skip_download,
                        insecure=args.insecure,
                        heavy_tasks=tasks,
                    )
                    print(path)
                    if args.with_notice:
                        linked = await attach_notices_async(
                            path,
                            Path(args.notice_out_root),
                            fetch_missing=not args.link_only,
                            sleep_s=args.sleep,
                            timeout=args.timeout,
                            insecure=args.insecure,
                            heavy_tasks=tasks,
                        )
                        for npath in linked:
                            print(npath)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    failed += 1
                    if _is_forbidden(exc):
                        await asyncio.sleep(max(args.sleep, FORBIDDEN_COOLDOWN_S[0]))
                        continue
                await asyncio.sleep(args.sleep)
        finally:
            await finish_heavy(tasks)
        return 1 if failed else 0

    return asyncio.run(_run())


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
    from ingest.notices.download import fetch_notice_async

    out_root = Path(args.out_root)
    numbers = _collect_numbers(args)
    if not numbers:
        print("Укажите --number <regNumber извещения>", file=sys.stderr)
        return 2

    async def _run() -> int:
        failed = 0
        tasks: list = []
        try:
            for num in numbers:
                print(f"Fetching notice {num} …", file=sys.stderr)
                try:
                    path = await fetch_notice_async(
                        num,
                        out_root,
                        sleep_s=args.sleep,
                        timeout=args.timeout,
                        skip_download=args.skip_download,
                        insecure=args.insecure,
                        kind=args.kind,
                        heavy_tasks=tasks,
                    )
                    print(path)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    failed += 1
                    if _is_forbidden(exc):
                        await asyncio.sleep(max(args.sleep, FORBIDDEN_COOLDOWN_S[0]))
                        continue
                await asyncio.sleep(args.sleep)
        finally:
            await finish_heavy(tasks)
        return 1 if failed else 0

    return asyncio.run(_run())


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
    common.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Пауза между жалобами и после ответа 403, сек. После успеха — 0,1 с",
    )
    common.add_argument("--timeout", type=float, default=15.0)
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
