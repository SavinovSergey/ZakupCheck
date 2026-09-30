"""Сборка NormUnit + structural Chunk из blocks.jsonl (DESIGN §4.2)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from schemas.corpus import Chunk, NormUnit

PART_RE = re.compile(r"^(\d+)\.\s+(.*)$", re.S)
POINT_RE = re.compile(r"^(\d+(?:\.\d+)?)\)\s+(.*)$", re.S)
SUBPOINT_RE = re.compile(r"^([а-яёa-z])\)\s+(.*)$", re.S | re.I)
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.?\s*(.*)$")

DEFAULT_MAX_CHUNK = 2000
DEFAULT_WINDOW_OVERLAP = 200


@dataclass
class _Seg:
    text: str
    char_start: int
    char_end: int


@dataclass
class _Point:
    number: str
    segs: list[_Seg] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segs)

    @property
    def char_start(self) -> int:
        return self.segs[0].char_start

    @property
    def char_end(self) -> int:
        return self.segs[-1].char_end


@dataclass
class _Part:
    number: str
    header: _Seg | None = None
    points: list[_Point] = field(default_factory=list)
    # текст части без пунктов (если пунктов нет) или «хвост»
    loose: list[_Seg] = field(default_factory=list)

    def full_text(self) -> str:
        parts: list[str] = []
        if self.header:
            parts.append(self.header.text)
        for p in self.points:
            parts.append(p.text)
        for s in self.loose:
            parts.append(s.text)
        return "\n".join(parts)

    def span(self) -> tuple[int, int]:
        starts: list[int] = []
        ends: list[int] = []
        if self.header:
            starts.append(self.header.char_start)
            ends.append(self.header.char_end)
        for p in self.points:
            starts.append(p.char_start)
            ends.append(p.char_end)
        for s in self.loose:
            starts.append(s.char_start)
            ends.append(s.char_end)
        if not starts:
            return 0, 0
        return min(starts), max(ends)


@dataclass
class _Article:
    number: str
    title: str | None
    title_seg: _Seg
    parts: list[_Part] = field(default_factory=list)
    # тело до первой части (редко)
    preamble: list[_Seg] = field(default_factory=list)


def make_norm_id(edition_id: str, article: str, part: str | None = None, point: str | None = None) -> str:
    nid = f"44FZ:{edition_id}:art.{article}"
    if part is not None:
        nid += f":ch.{part}"
    if point is not None:
        nid += f":p.{point}"
    return nid


def parse_articles(blocks: list[dict]) -> list[_Article]:
    articles: list[_Article] = []
    cur: _Article | None = None
    cur_part: _Part | None = None
    cur_point: _Point | None = None

    def finish_point() -> None:
        nonlocal cur_point
        cur_point = None

    def finish_part() -> None:
        nonlocal cur_part, cur_point
        cur_point = None
        cur_part = None

    for b in blocks:
        kind = b["kind"]
        text = b["text"]
        seg = _Seg(text=text, char_start=b["char_start"], char_end=b["char_end"])

        if kind == "article_title":
            finish_part()
            m = ARTICLE_RE.match(text)
            if not m:
                continue
            cur = _Article(
                number=m.group(1),
                title=(m.group(2).strip() or None),
                title_seg=seg,
            )
            articles.append(cur)
            cur_part = None
            cur_point = None
            continue

        if kind == "chapter_title":
            finish_part()
            cur = None
            continue

        if kind != "body" or cur is None:
            continue

        if m := PART_RE.match(text):
            finish_point()
            cur_part = _Part(number=m.group(1), header=seg)
            cur.parts.append(cur_part)
            cur_point = None
            continue

        if m := POINT_RE.match(text):
            if cur_part is None:
                # пункт без объявленной части — синтетическая часть "0"
                cur_part = _Part(number="0")
                cur.parts.append(cur_part)
            cur_point = _Point(number=m.group(1), segs=[seg])
            cur_part.points.append(cur_point)
            continue

        if SUBPOINT_RE.match(text) or (cur_point is not None and not PART_RE.match(text)):
            # подпункты и продолжения → в текущий пункт (вариант B)
            if cur_point is not None:
                cur_point.segs.append(seg)
            elif cur_part is not None:
                cur_part.loose.append(seg)
            else:
                cur.preamble.append(seg)
            continue

        # прочий текст
        if cur_point is not None:
            cur_point.segs.append(seg)
        elif cur_part is not None:
            cur_part.loose.append(seg)
        else:
            cur.preamble.append(seg)

    return articles


def build_norm_units(edition_id: str, articles: list[_Article]) -> list[NormUnit]:
    units: list[NormUnit] = []
    order = 0

    for art in articles:
        art_id = make_norm_id(edition_id, art.number)
        # текст статьи = заголовок + все части (для get_norm_text уровня article)
        body_parts: list[str] = [art.title_seg.text]
        starts = [art.title_seg.char_start]
        ends = [art.title_seg.char_end]
        for seg in art.preamble:
            body_parts.append(seg.text)
            starts.append(seg.char_start)
            ends.append(seg.char_end)
        for part in art.parts:
            body_parts.append(part.full_text())
            a, b = part.span()
            starts.append(a)
            ends.append(b)
        art_text = "\n\n".join(body_parts)
        units.append(
            NormUnit(
                norm_id=art_id,
                edition_id=edition_id,
                article=art.number,
                level="article",
                parent_norm_id=None,
                title=art.title,
                text=art_text,
                char_start=min(starts),
                char_end=max(ends),
                order=order,
                is_retrieval_unit=False,
            )
        )
        order += 1

        for part in art.parts:
            part_id = make_norm_id(edition_id, art.number, part.number)
            p_start, p_end = part.span()
            p_text = part.full_text()
            units.append(
                NormUnit(
                    norm_id=part_id,
                    edition_id=edition_id,
                    article=art.number,
                    part=part.number,
                    level="part",
                    parent_norm_id=art_id,
                    title=None,
                    text=p_text,
                    char_start=p_start,
                    char_end=p_end,
                    order=order,
                    is_retrieval_unit=True,
                )
            )
            order += 1

            for pt in part.points:
                pt_id = make_norm_id(edition_id, art.number, part.number, pt.number)
                units.append(
                    NormUnit(
                        norm_id=pt_id,
                        edition_id=edition_id,
                        article=art.number,
                        part=part.number,
                        point=pt.number,
                        level="point",
                        parent_norm_id=part_id,
                        title=None,
                        text=pt.text,
                        char_start=pt.char_start,
                        char_end=pt.char_end,
                        order=order,
                        is_retrieval_unit=True,
                    )
                )
                order += 1

    return units


def sliding_windows(text: str, char_start: int, max_chunk: int, overlap: int) -> list[tuple[int, int, str]]:
    """Окна в координатах full_text (char_start — начало единицы)."""
    if len(text) <= max_chunk:
        return [(char_start, char_start + len(text), text)]
    step = max(max_chunk - overlap, 1)
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        j = min(i + max_chunk, len(text))
        # стараемся не рвать посередине слова: отодвинуть конец к пробелу
        if j < len(text):
            sp = text.rfind(" ", i + max_chunk // 2, j)
            if sp != -1:
                j = sp
        chunk = text[i:j].strip()
        if chunk:
            # точные оффсеты относительно исходного text
            local_start = text.find(chunk, i, j + 1)
            if local_start < 0:
                local_start = i
            local_end = local_start + len(chunk)
            out.append((char_start + local_start, char_start + local_end, chunk))
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


def format_chunk_text(body: str, article_title: str | None, use_prefix: bool) -> str:
    if use_prefix and article_title:
        return f"{article_title}.\n{body}"
    return body


def build_structural_chunks(
    edition_id: str,
    articles: list[_Article],
    *,
    max_chunk: int = DEFAULT_MAX_CHUNK,
    window_overlap: int = DEFAULT_WINDOW_OVERLAP,
    title_prefix: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0

    def add(
        chunking: str,
        body: str,
        char_start: int,
        char_end: int,
        norm_ids: list[str],
        article_title: str | None,
    ) -> None:
        nonlocal n
        n += 1
        chunks.append(
            Chunk(
                chunk_id=f"chk_{edition_id}_{n:05d}",
                edition_id=edition_id,
                chunking=chunking,  # type: ignore[arg-type]
                text=format_chunk_text(body, article_title, title_prefix),
                char_start=char_start,
                char_end=char_end,
                norm_ids=norm_ids,
                max_chunk=max_chunk,
                article_title_prefix=title_prefix,
            )
        )

    for art in articles:
        for part in art.parts:
            part_id = make_norm_id(edition_id, art.number, part.number)
            point_ids = [
                make_norm_id(edition_id, art.number, part.number, pt.number) for pt in part.points
            ]
            part_text = part.full_text()
            p_start, p_end = part.span()

            if len(part_text) <= max_chunk:
                add(
                    "part",
                    part_text,
                    p_start,
                    p_end,
                    [part_id, *point_ids],
                    art.title,
                )
                continue

            # длинная часть → по пунктам
            if part.points:
                for pt in part.points:
                    pt_id = make_norm_id(edition_id, art.number, part.number, pt.number)
                    if len(pt.text) <= max_chunk:
                        add("point", pt.text, pt.char_start, pt.char_end, [pt_id], art.title)
                    else:
                        for w_start, w_end, w_text in sliding_windows(
                            pt.text, pt.char_start, max_chunk, window_overlap
                        ):
                            add("window", w_text, w_start, w_end, [pt_id], art.title)
                # header + loose без пунктов: если остались и длинные — окна на part
                # (обычно пункты покрывают часть)
            else:
                # нет пунктов — окна по тексту части
                for w_start, w_end, w_text in sliding_windows(
                    part_text, p_start, max_chunk, window_overlap
                ):
                    add("window", w_text, w_start, w_end, [part_id], art.title)

    return chunks


def load_edition_dir(edition_dir: Path) -> tuple[dict, list[dict], str]:
    edition = json.loads((edition_dir / "edition.json").read_text(encoding="utf-8"))
    blocks = [json.loads(l) for l in (edition_dir / "blocks.jsonl").open(encoding="utf-8")]
    full_text = (edition_dir / "full_text.txt").read_text(encoding="utf-8")
    # full_text.txt ends with newline from download
    if full_text.endswith("\n"):
        full_text = full_text[:-1]
    return edition, blocks, full_text


def build_corpus(
    edition_dir: Path,
    *,
    max_chunk: int = DEFAULT_MAX_CHUNK,
    window_overlap: int = DEFAULT_WINDOW_OVERLAP,
    title_prefix: bool = False,
) -> dict:
    edition_meta, blocks, full_text = load_edition_dir(edition_dir)
    edition_id = edition_meta["edition_id"]
    articles = parse_articles(blocks)
    units = build_norm_units(edition_id, articles)
    chunks = build_structural_chunks(
        edition_id,
        articles,
        max_chunk=max_chunk,
        window_overlap=window_overlap,
        title_prefix=title_prefix,
    )

    # проверка оффсетов NormUnit против full_text (мягкая: текст должен быть substring)
    # жёсткая проверка: char slice ≈ unit.text может расходиться из‑за join "\n" vs "\n\n"
    # поэтому сверяем только что границы монотонны и в пределах файла
    n = len(full_text)
    for u in units:
        if not (0 <= u.char_start <= u.char_end <= n):
            raise AssertionError(f"bad span {u.norm_id}: {u.char_start}:{u.char_end} n={n}")

    units_path = edition_dir / "norm_units.jsonl"
    chunks_path = edition_dir / f"chunks_structural_m{max_chunk}.jsonl"
    # стабильный alias для дефолта
    alias_path = edition_dir / "chunks_structural.jsonl"

    with units_path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(u.model_dump_json() + "\n")

    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")

    if max_chunk == DEFAULT_MAX_CHUNK and not title_prefix:
        alias_path.write_text(chunks_path.read_text(encoding="utf-8"), encoding="utf-8")

    stats = {
        "edition_id": edition_id,
        "articles": len(articles),
        "norm_units": len(units),
        "by_level": {
            "article": sum(1 for u in units if u.level == "article"),
            "part": sum(1 for u in units if u.level == "part"),
            "point": sum(1 for u in units if u.level == "point"),
        },
        "chunks": len(chunks),
        "chunking": {
            "part": sum(1 for c in chunks if c.chunking == "part"),
            "point": sum(1 for c in chunks if c.chunking == "point"),
            "window": sum(1 for c in chunks if c.chunking == "window"),
        },
        "max_chunk": max_chunk,
        "title_prefix": title_prefix,
        "units_path": str(units_path),
        "chunks_path": str(chunks_path),
    }
    (edition_dir / f"corpus_stats_m{max_chunk}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Собрать NormUnit и structural chunks из data/law/{edition_id}/"
    )
    parser.add_argument(
        "--edition-dir",
        type=Path,
        default=Path("data/law/2026-08-04"),
        help="Каталог редакции с blocks.jsonl и edition.json",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=DEFAULT_MAX_CHUNK,
        help=f"Порог структурного чанка в символах (старт эксперимента: {DEFAULT_MAX_CHUNK})",
    )
    parser.add_argument("--window-overlap", type=int, default=DEFAULT_WINDOW_OVERLAP)
    parser.add_argument(
        "--title-prefix",
        action="store_true",
        help='Добавить заголовок статьи в Chunk.text: "{title}.\\n{body}"',
    )
    args = parser.parse_args(argv)

    if not (args.edition_dir / "blocks.jsonl").exists():
        print(f"нет blocks.jsonl в {args.edition_dir}", file=sys.stderr)
        return 1

    stats = build_corpus(
        args.edition_dir,
        max_chunk=args.max_chunk,
        window_overlap=args.window_overlap,
        title_prefix=args.title_prefix,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
