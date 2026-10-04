"""Извлечение текста из pdf / docx / odt / html / txt / zip / изображений.

OCR (tesseract + pdftoppm) — fallback для сканов без текстового слоя.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
# mammoth экранирует пунктуацию для Markdown; для сырого корпуса снимаем.
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!])")
# подписи/сканы в docx → data:image base64 раздувают txt без пользы для eval
_MD_DATA_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image\/[^)]+\)", re.I)

# если pypdf дал меньше — считаем скан и идём в OCR
_PDF_TEXT_MIN_CHARS = 40

IMAGE_SUFFIXES = frozenset({"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"})
ARCHIVE_SUFFIXES = frozenset({"zip"})


def sniff_format(path: Path) -> str | None:
    """Определить тип файла по magic bytes (ЕИС часто врёт в имени/Content-Type)."""
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        # zip-контейнер: docx / odt / обычный zip
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return "zip"
        if "word/document.xml" in names:
            return "docx"
        if "content.xml" in names and any(
            n.startswith("META-INF/") or n == "mimetype" for n in names
        ):
            return "odt"
        return "zip"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] in {b"II*\x00", b"MM\x00*"}:
        return "tif"
    # OLE Compound File (legacy .doc / .xls / …)
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        # по расширению различим; без него чаще всего doc в выгрузках ФАС
        ext = path.suffix.lower().lstrip(".")
        if ext in {"xls", "xlsx"}:
            return "xls"
        return "doc"
    low = head.lstrip()[:20].lower()
    if low.startswith(b"<!doctype html") or low.startswith(b"<html"):
        return "html"
    if low.startswith(b"{\\rtf"):
        return "rtf"
    return None


def correct_suffix(path: Path) -> Path:
    """Если расширение не совпадает с содержимым — вернуть путь с правильным суффиксом."""
    kind = sniff_format(path)
    if not kind:
        return path
    want = f".{kind}"
    if path.suffix.lower() == want:
        return path
    target = path.with_suffix(want)
    if target.exists() and target != path:
        # не затирать: stem_1.ext
        i = 1
        while target.exists():
            target = path.with_name(f"{path.stem}_{i}{want}")
            i += 1
    path.rename(target)
    return target


def unescape_markdown(text: str) -> str:
    text = _MD_DATA_IMAGE_RE.sub("", text)
    return _MD_ESCAPE_RE.sub(r"\1", text)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NL_RE.sub("\n\n", text).strip()


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        elif tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_text("".join(self._parts))


def extract_html(data: bytes | str) -> str:
    if isinstance(data, bytes):
        for enc in ("utf-8", "windows-1251", "cp1251"):
            try:
                raw = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raw = data.decode("utf-8", errors="replace")
    else:
        raw = data
    parser = _HTMLText()
    parser.feed(raw)
    parser.close()
    return parser.text()


def extract_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "windows-1251", "cp1251"):
        try:
            return normalize_text(data.decode(enc))
        except UnicodeDecodeError:
            continue
    return normalize_text(data.decode("utf-8", errors="replace"))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _element_plain_text(el: ET.Element) -> str:
    """Собрать текст узла без лишних пробелов между children (OOXML/ODT)."""
    parts: list[str] = []

    def walk(node: ET.Element, *, root: bool) -> None:
        name = _local(node.tag)
        if not root and name in {"p", "h", "list-item"}:
            return
        if name in {"tab"}:
            parts.append("\t")
        elif name in {"br", "line-break"}:
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child, root=False)
            if child.tail:
                parts.append(child.tail)

    walk(el, root=True)
    return "".join(parts).replace("\u00a0", " ")


def _xml_texts(root: ET.Element) -> str:
    """Fallback OOXML/ODT: абзацы целиком, без пробелов между runs."""
    lines: list[str] = []
    for el in root.iter():
        if _local(el.tag) not in {"p", "h", "list-item"}:
            continue
        line = _element_plain_text(el)
        if line.strip():
            lines.append(line)
    return normalize_text("\n".join(lines))


def _extract_docx_ooxml(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    return _xml_texts(ET.fromstring(xml))


def extract_docx(path: Path) -> str:
    """DOCX → Markdown через mammoth; иначе абзацный OOXML-fallback."""
    try:
        import mammoth  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Для DOCX нужен пакет mammoth: pip install 'zakup-check[fas]' или pip install mammoth"
        ) from exc

    try:
        with path.open("rb") as fh:
            result = mammoth.convert_to_markdown(fh)
        text = (result.value or "").strip()
    except Exception:
        text = ""
    if text:
        text = unescape_markdown(text)
        text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
        return _NL_RE.sub("\n\n", text).strip()
    return _extract_docx_ooxml(path)


def extract_odt(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("content.xml")
    root = ET.fromstring(xml)
    return _xml_texts(root)


def _require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Для OCR нужен бинарь `{name}` "
            f"(обычно пакеты tesseract-ocr / poppler-utils)"
        )
    return path


def ocr_image(path: Path, *, lang: str = "rus+eng") -> str:
    """OCR одного изображения через tesseract."""
    tesseract = _require_bin("tesseract")
    proc = subprocess.run(
        [tesseract, str(path), "stdout", "-l", lang, "--psm", "3"],
        check=False,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"tesseract failed for {path.name}: {err or proc.returncode}")
    return normalize_text(proc.stdout.decode("utf-8", errors="replace"))


def ocr_pdf(path: Path, *, lang: str = "rus+eng", dpi: int = 200) -> str:
    """PDF → PNG (pdftoppm) → tesseract. Нужны poppler-utils и tesseract."""
    pdftoppm = _require_bin("pdftoppm")
    _require_bin("tesseract")
    with tempfile.TemporaryDirectory(prefix="zakupcheck-ocr-") as tmp:
        tmp_dir = Path(tmp)
        prefix = tmp_dir / "page"
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), str(path), str(prefix)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        pages = sorted(tmp_dir.glob("page-*.png"))
        if not pages:
            return ""
        parts: list[str] = []
        for page in pages:
            parts.append(ocr_image(page, lang=lang))
        return normalize_text("\n\n".join(p for p in parts if p))


def extract_pdf(path: Path, *, ocr_fallback: bool = True) -> tuple[str, str]:
    """PDF: сначала текстовый слой (pypdf), иначе OCR. Возвращает (text, method)."""
    kind = sniff_format(path)
    if kind and kind != "pdf":
        # ЕИС: «…из внешней системы.pdf», а внутри docx/zip
        if kind == "docx":
            return extract_docx(path), "docx-md"
        if kind == "odt":
            return extract_odt(path), "odt"
        if kind == "zip":
            return "", "archive:zip"
        return "", f"unsupported:misnamed-pdf-as-{kind}"

    if kind is None:
        # не похоже на PDF — не кормить pypdf (шум в stderr: invalid pdf header)
        return "", "unsupported:not-pdf"

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Для PDF нужен пакет pypdf: pip install 'zakup-check[fas]' или pip install pypdf"
        ) from exc
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:  # noqa: BLE001
        if ocr_fallback:
            try:
                ocr_text = ocr_pdf(path)
                if ocr_text:
                    return ocr_text, "pdf-ocr"
            except Exception:
                pass
        raise RuntimeError(f"битый PDF: {exc}") from exc
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = normalize_text("\n\n".join(parts))
    if len(text) >= _PDF_TEXT_MIN_CHARS or not ocr_fallback:
        return text, "pdf"
    try:
        ocr_text = ocr_pdf(path)
    except Exception:
        ocr_text = ""
    if ocr_text:
        return ocr_text, "pdf-ocr"
    return text, "pdf"


def _libreoffice_bin() -> str:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "Для .doc/.rtf нужен LibreOffice (soffice). "
        "Установите libreoffice-writer или конвертируйте файл в .docx вручную."
    )


def convert_with_libreoffice(path: Path, *, to_ext: str = "docx") -> Path:
    """Конвертировать документ через LibreOffice headless → временный файл."""
    soffice = _libreoffice_bin()
    tmp_root = Path(tempfile.mkdtemp(prefix="zakupcheck-lo-"))
    profile = tmp_root / "profile"
    profile.mkdir()
    out_dir = tmp_root / "out"
    out_dir.mkdir()
    # отдельный UserInstallation — иначе soffice падает на блокировке профиля
    env_user = f"-env:UserInstallation=file://{profile.resolve()}"
    try:
        proc = subprocess.run(
            [
                soffice,
                "--headless",
                "--norestore",
                "--nolockcheck",
                env_user,
                "--convert-to",
                to_ext,
                "--outdir",
                str(out_dir),
                str(path.resolve()),
            ],
            check=False,
            capture_output=True,
            timeout=180,
        )
        converted = list(out_dir.glob(f"*.{to_ext}"))
        if proc.returncode != 0 or not converted:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LibreOffice не смог конвертировать {path.name} → .{to_ext}: "
                f"{err.strip() or proc.returncode}"
            )
        # вернём путь; tmp_root живёт, пока не удалим после чтения
        dest = tmp_root / f"converted.{to_ext}"
        shutil.move(str(converted[0]), dest)
        return dest
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise


def extract_doc(path: Path) -> str:
    """Legacy .doc / .rtf → DOCX (LibreOffice) → Markdown (mammoth)."""
    converted = convert_with_libreoffice(path, to_ext="docx")
    try:
        return extract_docx(converted)
    finally:
        # convert_with_libreoffice кладёт файл в mkdtemp-корень
        shutil.rmtree(converted.parent, ignore_errors=True)


def safe_unpack_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Распаковать zip без zip-slip. Возвращает пути к файлам (не каталогам)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            # нормализуем и отсекаем абсолютные / ..
            target = (dest_dir / name).resolve()
            if not str(target).startswith(str(dest_root) + "/") and target != dest_root:
                raise RuntimeError(f"zip-slip: опасный путь в архиве: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def extract_file(path: Path, *, ocr_fallback: bool = True) -> tuple[str, str]:
    """Возвращает (text, method). method = ext / pdf-ocr / archive:zip / unsupported."""
    # Сначала содержимое, потом расширение (ЕИС часто врёт в имени).
    kind = sniff_format(path)
    suffix = (kind or path.suffix.lower().lstrip("."))

    if suffix in {"html", "htm", "xhtml"}:
        return extract_html(path.read_bytes()), "html"
    if suffix in {"txt", "text", "csv", "log"}:
        return extract_txt(path.read_bytes()), "txt"
    if suffix == "docx":
        return extract_docx(path), "docx-md"
    if suffix == "odt":
        return extract_odt(path), "odt"
    if suffix in {"doc", "rtf"}:
        return extract_doc(path), "doc-lo"
    if suffix == "pdf":
        return extract_pdf(path, ocr_fallback=ocr_fallback)
    if suffix in IMAGE_SUFFIXES:
        if not ocr_fallback:
            return "", f"unsupported:{suffix}"
        return ocr_image(path), "image-ocr"
    if suffix in ARCHIVE_SUFFIXES:
        return "", f"archive:{suffix}"
    if suffix in {"xlsx", "xls"}:
        return "", f"unsupported:{suffix}"
    try:
        return extract_txt(path.read_bytes()), "txt-fallback"
    except Exception:
        return "", f"unsupported:{suffix or 'unknown'}"
