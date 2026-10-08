#!/usr/bin/env python3
"""Собрать новые номера жалоб ЕИС (44-ФЗ, жалоба признана обоснованной).

Список карточек берётся со страницы results.html, не с формы search_eis.html.
Уже записанные в файл номеров строки пропускаются.

Пример (сертификат ЕИС с этой машины не проверяется, как у --insecure у загрузчика):

  python3 scripts/collect_complaint_numbers.py --insecure
  python3 scripts/collect_complaint_numbers.py --insecure --pages 5 --append
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fas.download import decode_html, http_get  # noqa: E402

EIS_RESULTS = "https://zakupki.gov.ru/epz/complaint/search/results.html"
NUMBER_RE = re.compile(r"complaintNumber=([^&\"'\s<>]+)", re.I)
QUEUE_MARK = "очередь"


def load_known(path: Path) -> set[str]:
    known: set[str] = set()
    if not path.is_file():
        return known
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            known.add(line)
    return known


def results_url(*, page: int, per_page: int) -> str:
    # Те же фильтры, что в ручном поиске: 44-ФЗ, результат «обоснована»,
    # сортировка по дате обновления, новые сверху.
    return (
        f"{EIS_RESULTS}?morphology=on"
        "&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D0%BE%D0%B1%D0%BD%D0%BE%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F"
        "&fz94=on&decisionOnTheComplaintTypeResult_0=on"
        "&decisionOnTheComplaintTypeResult=0"
        "&sortBy=UPDATE_DATE&sortDirection=false"
        f"&recordsPerPage=_{per_page}&showLotsInfoHidden=false"
        f"&pageNumber={page}"
    )


def collect(
    *,
    pages: int,
    per_page: int,
    known: set[str],
    insecure: bool,
    timeout: float,
) -> list[str]:
    found: list[str] = []
    seen = set(known)
    for page in range(1, pages + 1):
        data, _ctype, _url = http_get(
            results_url(page=page, per_page=per_page),
            timeout=timeout,
            retries=2,
            sleep_s=0.5,
            insecure=insecure,
        )
        html = decode_html(data)
        nums = NUMBER_RE.findall(html)
        fresh = 0
        for num in nums:
            if num in seen:
                continue
            seen.add(num)
            found.append(num)
            fresh += 1
        print(f"page {page}: {len(nums)} на странице, новых {fresh}", file=sys.stderr)
    return found


def append_queue(path: Path, numbers: list[str]) -> None:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    if QUEUE_MARK not in text:
        text += f"\n# --- {QUEUE_MARK} ---\n"
    text += "".join(f"{num}\n" for num in numbers)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать новые номера жалоб ЕИС")
    parser.add_argument("--pages", type=int, default=5, help="Сколько страниц выдачи, с 1-й")
    parser.add_argument("--per-page", type=int, default=20, choices=(10, 20, 50))
    parser.add_argument(
        "--numbers-file",
        type=Path,
        default=Path("evals/numbers_fas.txt"),
        help="Уже известные номера; с --append сюда же дописывается очередь",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Дописать новые номера в конец файла, не только напечатать",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Не проверять сертификат (на этой машине у ЕИС самоподписанная цепочка)",
    )
    args = parser.parse_args(argv)

    known = load_known(args.numbers_file)
    found = collect(
        pages=args.pages,
        per_page=args.per_page,
        known=known,
        insecure=args.insecure,
        timeout=args.timeout,
    )
    print(f"новых: {len(found)}", file=sys.stderr)
    for num in found:
        print(num)
    if args.append and found:
        append_queue(args.numbers_file, found)
        print(f"дописано в {args.numbers_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
