#!/usr/bin/env python3
"""Пакетная загрузка жалоб ЕИС (+ опционально извещений) для разметки eval.

Пример:
  python scripts/batch_download_fas.py \\
    --numbers-file evals/numbers_fas.txt --with-notice --insecure

Файл номеров: по одному complaintNumber на строку, # — комментарий.
Уже скачанные (есть meta.json) пропускаются при --skip-existing (по умолчанию).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fas.download import (  # noqa: E402
    attach_notices,
    fetch_complaint,
)


def load_numbers(path: Path) -> list[str]:
    numbers: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # допускаем URL с complaintNumber=
        if "complaintNumber=" in line:
            from ingest.fas.download import COMPLAINT_NUM_RE

            m = COMPLAINT_NUM_RE.search(line)
            if not m:
                print(f"skip (нет complaintNumber): {line}", file=sys.stderr)
                continue
            line = m.group(1)
        if line in seen:
            continue
        seen.add(line)
        numbers.append(line)
    return numbers


def complaint_ready(out_dir: Path) -> bool:
    return (out_dir / "meta.json").exists()


def summarize_dir(out_dir: Path) -> dict:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return {"complaint_dir": str(out_dir), "has_meta": False}
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    extracted = raw.get("extracted") or []
    return {
        "complaint_dir": str(out_dir),
        "has_meta": True,
        "procurement_ids": raw.get("procurement_ids") or [],
        "notice_dirs": raw.get("notice_dirs") or [],
        "documents": len(raw.get("documents") or []),
        "extracted": len(extracted),
        "extract_methods": sorted(
            {e.get("method") for e in extracted if e.get("method")}
        ),
        "notes": raw.get("notes") or [],
    }


def write_report(rows: list[dict], report_jsonl: Path, report_csv: Path) -> None:
    report_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with report_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "complaint_number",
        "status",
        "complaint_dir",
        "procurement_ids",
        "notice_dirs",
        "documents",
        "extracted",
        "extract_methods",
        "error",
        "elapsed_s",
    ]
    with report_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("procurement_ids", "notice_dirs", "extract_methods", "notes"):
                if key in flat and isinstance(flat[key], list):
                    flat[key] = ";".join(str(x) for x in flat[key])
            w.writerow(flat)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Пакетно скачать жалобы ЕИС (+ извещения) для eval"
    )
    p.add_argument(
        "--numbers-file",
        type=Path,
        required=True,
        help="Файл с номерами жалоб (по одному на строку)",
    )
    p.add_argument("--out-root", type=Path, default=Path("data/raw/fas"))
    p.add_argument("--notice-out-root", type=Path, default=Path("data/raw/notices"))
    p.add_argument(
        "--with-notice",
        action="store_true",
        default=True,
        help="Связать/скачать извещения (по умолчанию вкл.)",
    )
    p.add_argument(
        "--no-with-notice",
        action="store_false",
        dest="with_notice",
        help="Только жалобы, без извещений",
    )
    p.add_argument(
        "--link-only",
        action="store_true",
        help="Не докачивать отсутствующие извещения, только связать",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Пропускать жалобы, у которых уже есть meta.json (по умолчанию)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Перекачать даже если meta.json уже есть",
    )
    p.add_argument("--skip-download", action="store_true", help="Только HTML, без вложений")
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--insecure", action="store_true")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("evals/batch_reports"),
        help="Куда писать отчёт jsonl/csv",
    )
    args = p.parse_args(argv)

    if args.force:
        args.skip_existing = False

    numbers = load_numbers(args.numbers_file)
    if not numbers:
        print(f"В {args.numbers_file} нет номеров", file=sys.stderr)
        return 2

    print(
        f"Batch: {len(numbers)} жалоб → {args.out_root} "
        f"(with_notice={args.with_notice}, skip_existing={args.skip_existing})",
        file=sys.stderr,
    )

    rows: list[dict] = []
    ok = skip = fail = 0
    started = datetime.now(timezone.utc).isoformat()

    for i, num in enumerate(numbers, 1):
        out_dir = args.out_root / num
        print(f"[{i}/{len(numbers)}] {num} …", file=sys.stderr)
        t0 = time.time()
        row: dict = {
            "complaint_number": num,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if args.skip_existing and complaint_ready(out_dir):
            row["status"] = "skipped"
            row.update(summarize_dir(out_dir))
            if args.with_notice:
                try:
                    linked = attach_notices(
                        out_dir,
                        args.notice_out_root,
                        fetch_missing=not args.link_only,
                        sleep_s=args.sleep,
                        timeout=args.timeout,
                        insecure=args.insecure,
                    )
                    row["notice_dirs"] = [str(p) for p in linked] or row.get(
                        "notice_dirs"
                    )
                    row["status"] = "skipped_linked"
                except RuntimeError as exc:
                    row["link_error"] = str(exc)
            row["elapsed_s"] = round(time.time() - t0, 2)
            rows.append(row)
            skip += 1
            print(f"  skip existing → {out_dir}", file=sys.stderr)
            continue

        try:
            path = fetch_complaint(
                num,
                args.out_root,
                sleep_s=args.sleep,
                timeout=args.timeout,
                skip_download=args.skip_download,
                insecure=args.insecure,
            )
            row.update(summarize_dir(path))
            row["status"] = "ok"
            if args.with_notice:
                linked = attach_notices(
                    path,
                    args.notice_out_root,
                    fetch_missing=not args.link_only,
                    sleep_s=args.sleep,
                    timeout=args.timeout,
                    insecure=args.insecure,
                )
                row["notice_dirs"] = [str(p) for p in linked]
                row.update(summarize_dir(path))
                row["status"] = "ok"
            ok += 1
            print(f"  ok → {path}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            row.update(summarize_dir(out_dir))
            fail += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

        row["elapsed_s"] = round(time.time() - t0, 2)
        rows.append(row)
        time.sleep(args.sleep)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_jsonl = args.report_dir / f"batch_{stamp}.jsonl"
    report_csv = args.report_dir / f"batch_{stamp}.csv"
    write_report(rows, report_jsonl, report_csv)

    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "numbers_file": str(args.numbers_file),
        "total": len(numbers),
        "ok": ok,
        "skipped": skip,
        "failed": fail,
        "report_jsonl": str(report_jsonl),
        "report_csv": str(report_csv),
    }
    (args.report_dir / f"batch_{stamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Готово: ok={ok} skipped={skip} failed={fail}\n"
        f"Отчёт: {report_csv}",
        file=sys.stderr,
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
