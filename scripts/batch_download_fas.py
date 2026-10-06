#!/usr/bin/env python3
"""Пакетная загрузка жалоб ЕИС (+ опционально извещений) для разметки eval.

Пример:
  python scripts/batch_download_fas.py \\
    --numbers-file evals/numbers_fas.txt --with-notice --insecure

Файл номеров: по одному complaintNumber на строку, # — комментарий.
Уже скачанные (есть HTML-текст карточки) пропускаются при
--skip-existing (по умолчанию).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fas.download import (  # noqa: E402
    COMPLAINT_NUM_RE,
    _is_forbidden,
    attach_notices_async,
    fetch_complaint_async,
    finish_heavy,
)


def load_numbers(path: Path) -> list[str]:
    numbers: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Допускаем URL с complaintNumber=, но не пропускаем
        # произвольные строки в URL и имена каталогов.
        m = COMPLAINT_NUM_RE.search(line)
        if m:
            line = m.group(1)
        elif not line.isascii() or not line.isdigit():
            print(f"skip (неверный complaintNumber): {line}", file=sys.stderr)
            continue
        if line in seen:
            continue
        seen.add(line)
        numbers.append(line)
    return numbers


def complaint_ready(out_dir: Path) -> bool:
    """Готова только жалоба с текстом карточки. Пустой meta после 403 берём снова."""
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    for entry in raw.get("extracted") or []:
        if not isinstance(entry, dict) or entry.get("method") != "html":
            continue
        text_path = entry.get("text_path")
        if isinstance(text_path, str) and (out_dir / text_path).is_file():
            return True
    return False


def summarize_dir(out_dir: Path) -> dict:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return {"complaint_dir": str(out_dir), "has_meta": False}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("meta.json должен содержать JSON-объект")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "complaint_dir": str(out_dir),
            "has_meta": True,
            "meta_error": str(exc),
        }
    extracted = [e for e in raw.get("extracted") or [] if isinstance(e, dict)]
    documents = [d for d in raw.get("documents") or [] if isinstance(d, dict)]
    document_errors = [
        f"{d.get('title') or d.get('url') or '?'}: {d.get('error') or 'ошибка скачивания'}"
        for d in documents
        if d.get("status") == "error"
    ]
    extract_errors = [
        f"{e.get('source') or '?'}: {e.get('error') or 'ошибка извлечения'}"
        for e in extracted
        if e.get("method") == "error"
    ]
    return {
        "complaint_dir": str(out_dir),
        "has_meta": True,
        "procurement_ids": raw.get("procurement_ids") or [],
        "notice_dirs": raw.get("notice_dirs") or [],
        "documents": len(documents),
        "document_errors": document_errors,
        "extracted": len(extracted),
        "extract_methods": sorted(
            {e.get("method") for e in extracted if e.get("method")}
        ),
        "extract_errors": extract_errors,
        "notes": raw.get("notes") or [],
    }


def collect_notice_errors(notice_dirs: list[object]) -> list[str]:
    """Собрать ошибки скачивания/извлечения из связанных извещений."""
    errors: list[str] = []
    for raw_dir in notice_dirs:
        meta_path = Path(str(raw_dir)) / "meta.json"
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("meta.json должен содержать JSON-объект")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{meta_path}: {exc}")
            continue
        for doc in raw.get("documents") or []:
            if isinstance(doc, dict) and doc.get("status") == "error":
                label = doc.get("title") or doc.get("url") or "?"
                errors.append(f"{meta_path.parent.name}/{label}: {doc.get('error')}")
        for entry in raw.get("extracted") or []:
            if isinstance(entry, dict) and entry.get("method") == "error":
                label = entry.get("source") or "?"
                errors.append(f"{meta_path.parent.name}/{label}: {entry.get('error')}")
    return errors


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
        "document_errors",
        "extracted",
        "extract_methods",
        "extract_errors",
        "notice_errors",
        "meta_error",
        "notes",
        "error",
        "elapsed_s",
    ]
    with report_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            flat = dict(row)
            for key in (
                "procurement_ids",
                "notice_dirs",
                "extract_methods",
                "document_errors",
                "extract_errors",
                "notice_errors",
                "notes",
            ):
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
    p.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Пауза между повторами и после ошибки, сек. После успеха паузы нет",
    )
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument(
        "--max-pending-heavy",
        type=int,
        default=4,
        help="Порог фоновых OCR/LibreOffice-задач для сбора между жалобами",
    )
    p.add_argument("--insecure", action="store_true")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("evals/batch_reports"),
        help="Куда писать отчёт jsonl/csv",
    )
    args = p.parse_args(argv)

    if args.sleep < 0 or args.timeout <= 0 or args.max_pending_heavy <= 0:
        p.error("--sleep должен быть >= 0, --timeout и --max-pending-heavy — > 0")

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
    heavy_errors: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_jsonl = args.report_dir / f"batch_{stamp}.jsonl"
    report_csv = args.report_dir / f"batch_{stamp}.csv"

    async def _run() -> None:
        tasks: list = []
        try:
            await _each(tasks)
        finally:
            heavy_errors.extend(await finish_heavy(tasks))

    async def _each(tasks: list) -> None:
        for i, num in enumerate(numbers, 1):
            out_dir = args.out_root / num
            print(f"[{i}/{len(numbers)}] {num} …", file=sys.stderr)
            t0 = time.perf_counter()
            row: dict = {
                "complaint_number": num,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            complaint_downloaded = complaint_ready(out_dir)
            try:
                if args.skip_existing and complaint_downloaded:
                    row["status"] = "skipped"
                    row.update(summarize_dir(out_dir))
                    if args.with_notice:
                        linked = await attach_notices_async(
                            out_dir,
                            args.notice_out_root,
                            fetch_missing=not args.link_only,
                            sleep_s=args.sleep,
                            timeout=args.timeout,
                            insecure=args.insecure,
                            heavy_tasks=tasks,
                        )
                        row["notice_dirs"] = [str(p) for p in linked] or row.get(
                            "notice_dirs"
                        )
                        row["status"] = "skipped_linked"
                    print(f"  skip existing → {out_dir}", file=sys.stderr)
                else:
                    path = await fetch_complaint_async(
                        num,
                        args.out_root,
                        sleep_s=args.sleep,
                        timeout=args.timeout,
                        skip_download=args.skip_download,
                        insecure=args.insecure,
                        heavy_tasks=tasks,
                    )
                    row.update(summarize_dir(path))
                    row["status"] = "ok"
                    complaint_downloaded = True
                    if args.with_notice:
                        linked = await attach_notices_async(
                            path,
                            args.notice_out_root,
                            fetch_missing=not args.link_only,
                            sleep_s=args.sleep,
                            timeout=args.timeout,
                            insecure=args.insecure,
                            heavy_tasks=tasks,
                        )
                        row["notice_dirs"] = [str(p) for p in linked]
                        row.update(summarize_dir(path))
                    print(f"  ok → {path}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "partial" if complaint_downloaded else "error"
                row["error"] = str(exc)
                row.update(summarize_dir(out_dir))
                print(f"  ERROR: {exc}", file=sys.stderr)
            row["elapsed_s"] = round(time.perf_counter() - t0, 2)
            rows.append(row)

            if len(tasks) >= args.max_pending_heavy:
                heavy_errors.extend(await finish_heavy(tasks))

            # Внутри HTTP-клиента уже есть короткая пауза после успеха.
            # Здесь ждём только после ошибок; 403 уже имеет длинный backoff
            # в fetch_complaint_async/fetch_notice_async.
            if row["status"] in {"partial", "error"} and not _is_forbidden(
                RuntimeError(str(row.get("error") or ""))
            ):
                await asyncio.sleep(args.sleep)

    fatal_error: str | None = None
    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — отчёт нужен даже при сбое оркестратора
        fatal_error = str(exc)
        print(f"FATAL: {exc}", file=sys.stderr)

    for row in rows:
        complaint_dir = row.get("complaint_dir")
        if complaint_dir:
            row.update(summarize_dir(Path(complaint_dir)))
        if row.get("meta_error") and row["status"] not in {"error", "partial"}:
            row["status"] = "partial"
            row["error"] = f"meta.json: {row['meta_error']}"
        extract_errors = row.get("extract_errors") or []
        document_errors = row.get("document_errors") or []
        notice_errors = collect_notice_errors(row.get("notice_dirs") or [])
        row["notice_errors"] = notice_errors
        processing_errors = document_errors + extract_errors + notice_errors
        if processing_errors and row["status"] not in {"error", "partial"}:
            row["status"] = "partial"
            row["error"] = f"Ошибок файлов/извлечения: {len(processing_errors)}"

    if heavy_errors:
        by_dir = {str(row.get("complaint_dir")): row for row in rows}
        for item in heavy_errors:
            row = by_dir.get(str(item.get("out_dir")))
            if row and row["status"] not in {"error", "partial"}:
                row["status"] = "partial"
                row["error"] = f"heavy processing: {item.get('error')}"

    write_report(rows, report_jsonl, report_csv)

    ok = sum(row.get("status") == "ok" for row in rows)
    skip = sum(str(row.get("status", "")).startswith("skipped") for row in rows)
    partial = sum(row.get("status") == "partial" for row in rows)
    fail = sum(row.get("status") == "error" for row in rows)

    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "numbers_file": str(args.numbers_file),
        "total": len(numbers),
        "processed": len(rows),
        "unprocessed": len(numbers) - len(rows),
        "ok": ok,
        "skipped": skip,
        "partial": partial,
        "failed": fail,
        "fatal_error": fatal_error,
        "heavy_errors": heavy_errors,
        "report_jsonl": str(report_jsonl),
        "report_csv": str(report_csv),
    }
    (args.report_dir / f"batch_{stamp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Готово: ok={ok} skipped={skip} partial={partial} failed={fail}\n"
        f"Отчёт: {report_csv}",
        file=sys.stderr,
    )
    return 1 if fail or partial or fatal_error or heavy_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
