#!/usr/bin/env python3
"""Evaluate baseline E: an oracle issue-type to statutory-norm lookup."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.baseline_e import (  # noqa: E402
    MappingBuild,
    build_mapping,
    leave_one_group_out_predictions,
    overlay_mapping,
    predict_rows,
    score_predictions,
)
from index.error_analysis import paired_group_bootstrap  # noqa: E402
from index.recall import AggregateRecall, CaseScore, aggregate  # noqa: E402
from schemas.eval_case import VERIFIED_ISSUE_TYPES  # noqa: E402

SEED = 20261008
KS = (5, 10)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(metrics: AggregateRecall) -> dict[str, Any]:
    return {
        "n": metrics.n,
        "recall_exact": metrics.recall_exact,
        "all_exact": metrics.all_exact,
        "recall_article": metrics.recall_article,
        "all_article": metrics.all_article,
    }


def _score_at_k(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[str]],
    k: int,
) -> tuple[dict[str, Any], list[CaseScore]]:
    metrics, scores = score_predictions(rows, [tuple(item[:k]) for item in predictions])
    return _metrics(metrics), scores


def _confidence(scores: Sequence[CaseScore], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = [item.article.recall for item in scores]
    interval = paired_group_bootstrap(
        [0.0] * len(values),
        values,
        [str(row["case_group_id"]) for row in rows],
        seed=SEED,
    )
    return {
        "mean": interval.mean_delta,
        "low": interval.low,
        "high": interval.high,
    }


def _breakdown(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[str]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    result: list[dict[str, Any]] = []
    for value in values:
        indices = [index for index, row in enumerate(rows) if str(row[key]) == value]
        scores = [
            score_predictions([rows[index]], [predictions[index]])[1][0]
            for index in indices
        ]
        result.append(
            {
                key: value,
                **_metrics(aggregate(scores)),
                "status": "comparison" if len(indices) >= 5 else "descriptive",
            }
        )
    return result


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[str]],
) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    scores10: list[CaseScore] = []
    for k in KS:
        metrics, scores = _score_at_k(rows, predictions, k)
        by_k[f"at{k}"] = metrics
        if k == 10:
            scores10 = scores
    return {
        **by_k,
        "article_at10_ci": _confidence(scores10, rows),
        "by_issue_type": _breakdown(rows, predictions, key="issue_type"),
        "by_topic": _breakdown(rows, predictions, key="topic"),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("manifest_version", 0)) < 2:
        raise ValueError("baseline E requires retrieval manifest version >= 2")
    counts = manifest["counts"]
    for split in ("dev", "eval"):
        rows = manifest[split]
        if len(rows) != int(counts[split]):
            raise ValueError(f"manifest {split} count does not match rows")
        for row in rows:
            issue_type = row.get("issue_type")
            if issue_type not in VERIFIED_ISSUE_TYPES:
                raise ValueError(f"{row.get('case_id')}: invalid issue_type {issue_type!r}")


def _load_zero_shot(
    path: Path,
    *,
    edition_id: str,
    dev_rows: Sequence[Mapping[str, Any]],
    norm_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    payload = _load_json(path)
    if payload.get("edition_id") != edition_id:
        raise ValueError("zero-shot edition_id does not match manifest")
    overrides = payload.get("overrides", {})
    seen = {str(row["issue_type"]) for row in dev_rows}
    for issue_type, values in overrides.items():
        if issue_type not in VERIFIED_ISSUE_TYPES:
            raise ValueError(f"unknown zero-shot issue_type: {issue_type}")
        if issue_type in seen:
            raise ValueError(f"zero-shot override is not dev-unseen: {issue_type}")
        missing = set(values) - norm_ids
        if missing:
            raise ValueError(f"zero-shot norms are absent from corpus: {sorted(missing)}")
    return {key: tuple(values) for key, values in overrides.items()}


def _load_norm_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result.add(str(json.loads(line)["norm_id"]))
    return result


def _mapping_state(build: MappingBuild) -> dict[str, Any]:
    return {
        issue_type: [
            {
                "norm_id": norm_id,
                "group_support": build.group_support[issue_type][norm_id],
                "case_support": build.case_support[issue_type][norm_id],
            }
            for norm_id in norm_ids
        ]
        for issue_type, norm_ids in build.mapping.items()
    }


def _retriever_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    state = _load_json(path)
    return {
        "dev": state["winner"]["at10"],
        "eval": state.get("eval_reporting_snapshot", {}).get("winner"),
        "dev_precision": state.get("precision"),
        "eval_precision": state.get("eval_reporting_snapshot", {}).get("precision"),
    }


def _metric_row(label: str, split: str, result: Mapping[str, Any]) -> str:
    at5, at10 = result["at5"], result["at10"]
    ci = result["article_at10_ci"]
    return (
        f"| {split} | {label} | {at10['n']} | {at5['recall_exact']:.3f} | "
        f"{at5['recall_article']:.3f} | {at10['recall_exact']:.3f} | "
        f"{at10['all_exact']:.3f} | {at10['recall_article']:.3f} | "
        f"{at10['all_article']:.3f} | [{ci['low']:.3f}; {ci['high']:.3f}] |"
    )


def _breakdown_table(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    title = "issue_type" if key == "issue_type" else "topic"
    lines = [
        f"| {title} | n | exact R@10 | article R@10 | all article@10 | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row[key]}` | {row['n']} | {row['recall_exact']:.3f} | "
            f"{row['recall_article']:.3f} | {row['all_article']:.3f} | {row['status']} |"
        )
    return lines


def _report(state: Mapping[str, Any]) -> str:
    lines = [
        "# Baseline E: тип нарушения → нормы",
        "",
        "Baseline E не ищет по тексту закона. Он получает gold `issue_type` и возвращает "
        "упорядоченный список norm id для конкретной редакции закона. Поэтому это **oracle upper bound "
        "для классификации типа**, а не end-to-end метрика продукта.",
        "",
        "Основная оценка E-dev использует leave-one-`case_group_id`-out на dev. Затем full-dev таблица "
        "замораживается и применяется к eval. `other`, `unclear` и неизвестные типы намеренно не получают fallback.",
        "Baseline E не читает текст довода, поэтому raw и blind для него тождественны.",
        "",
        "## Агрегированные результаты",
        "",
        "Основная метрика: article R@10. Exact и all-hit остаются вторичными. Интервал — 95% "
        f"group bootstrap для article R@10, seed `{state['seed']}`.",
        "",
        "| split | method | n | exact R@5 | article R@5 | exact R@10 | all exact@10 | article R@10 | all article@10 | article R@10 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("E-dev group holdout", "dev", state["dev"]["group_holdout"]),
        _metric_row("E-dev apparent (diagnostic only)", "dev", state["dev"]["apparent"]),
    ]
    if "eval" in state:
        lines.extend(
            [
                _metric_row("E-dev frozen", "eval", state["eval"]["e_dev"]),
                _metric_row("E-dev frozen, dev-seen types", "eval", state["eval"]["seen_types"]),
                _metric_row("E-dev + zero-shot (post-hoc)", "eval", state["eval"]["zero_shot"]),
            ]
        )

    retriever = state.get("retriever_snapshot")
    if retriever:
        lines.extend(
            [
                "",
                "## Сравнение с замороженным retriever",
                "",
                "Метрики retriever взяты из уже замороженного state; cross-encoder не перезапускался. "
                "Парный интервал не приводится, потому что этот state хранит агрегаты eval, а не все покейсовые предсказания.",
                "",
                "| split | system | exact R@10 | article R@10 | all article@10 | precision |",
                "|---|---|---:|---:|---:|---|",
                f"| dev | frozen hybrid + CE | {retriever['dev']['recall_exact']:.3f} | "
                f"{retriever['dev']['recall_article']:.3f} | {retriever['dev']['all_article']:.3f} | "
                f"{retriever['dev_precision']} |",
            ]
        )
        if retriever.get("eval"):
            lines.append(
                f"| eval | frozen hybrid + CE snapshot | {retriever['eval']['recall_exact']:.3f} | "
                f"{retriever['eval']['recall_article']:.3f} | {retriever['eval']['all_article']:.3f} | "
                f"{retriever['eval_precision']} |"
            )

    lines.extend(["", "## Замороженная E-dev таблица", ""])
    lines.extend(
        [
            "| issue_type | ordered norms (group support / case support) |",
            "|---|---|",
        ]
    )
    for issue_type, norms in state["mapping"].items():
        rendered = "; ".join(
            f"`{item['norm_id']}` ({item['group_support']}/{item['case_support']})"
            for item in norms
        )
        lines.append(f"| `{issue_type}` | {rendered} |")
    lines.append("| `other`, `unclear`, unknown | no fallback |")

    if "eval" in state:
        lines.extend(
            [
                "",
                "## Zero-shot sensitivity",
                "",
                "Два issue type встречаются только в eval. Overlay ниже показан отдельно и не может "
                "называться слепым, так как eval manifest уже был просмотрен:",
                "",
            ]
        )
        for issue_type, norms in state["zero_shot_overrides"].items():
            lines.append(f"- `{issue_type}`: {', '.join(f'`{item}`' for item in norms)}")

    lines.extend(
        [
            "",
            "## Dev group-holdout по issue type",
            "",
            "Строки с n < 5 считаются только описательными.",
            "",
            *_breakdown_table(state["dev"]["group_holdout"]["by_issue_type"], "issue_type"),
            "",
            "## Dev group-holdout по темам",
            "",
            *_breakdown_table(state["dev"]["group_holdout"]["by_topic"], "topic"),
        ]
    )
    if "eval" in state:
        lines.extend(
            [
                "",
                "## Eval E-dev по issue type",
                "",
                *_breakdown_table(state["eval"]["e_dev"]["by_issue_type"], "issue_type"),
            ]
        )

    dev = state["dev"]["group_holdout"]["at10"]
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            f"При oracle issue type защищённая от group leakage dev-оценка E-dev даёт article R@10 `{dev['recall_article']:.3f}` "
            f"и exact R@10 `{dev['recall_exact']:.3f}`. Это измеряет ценность таблицы после классификации, "
            "а не способность найти нарушение в документе.",
        ]
    )
    if "eval" in state:
        evaluated = state["eval"]["e_dev"]["at10"]
        zero = state["eval"]["zero_shot"]["at10"]
        lines.append(
            f"Замороженная E-dev таблица даёт на eval article/exact R@10 "
            f"`{evaluated['recall_article']:.3f}/{evaluated['recall_exact']:.3f}`. Post-hoc zero-shot overlay "
            f"изменяет их до `{zero['recall_article']:.3f}/{zero['recall_exact']:.3f}`."
        )
    lines.extend(
        [
            "Retrieval остаётся нужен для неизвестных типов, выбора evidence и случаев, где upstream-детектор "
            "не уверен. Следующий end-to-end эксперимент должен заменить gold issue type на предсказание эвристики/LLM.",
            "",
            "Machine-readable state: `data/analysis/baseline_e_state.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _merge_aggregate(
    state: Mapping[str, Any],
    *,
    aggregate_state_path: Path,
    aggregate_report_path: Path,
    report_path: Path,
) -> None:
    if not aggregate_state_path.is_file():
        return
    aggregate = _load_json(aggregate_state_path)
    aggregate["baseline_e"] = {
        "version": state["version"],
        "oracle_issue_type": state["oracle_issue_type"],
        "dev": state["dev"],
        "mapping": state["mapping"],
        "report": str(report_path),
        **({"eval": state["eval"]} if "eval" in state else {}),
        **(
            {"zero_shot_overrides": state["zero_shot_overrides"]}
            if "zero_shot_overrides" in state
            else {}
        ),
    }
    aggregate_state_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(aggregate, aggregate_report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline E from the frozen retrieval manifest")
    parser.add_argument("--manifest", type=Path, default=Path("evals/retrieval_manifest.json"))
    parser.add_argument(
        "--zero-shot",
        type=Path,
        default=Path("evals/baselines/baseline_e_zero_shot.json"),
    )
    parser.add_argument(
        "--norms",
        type=Path,
        default=Path("data/law/2026-08-04/norm_units.jsonl"),
    )
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--report", type=Path, default=Path("evals/reports/baseline_e.md"))
    parser.add_argument("--state", type=Path, default=Path("data/analysis/baseline_e_state.json"))
    parser.add_argument(
        "--aggregate-state",
        type=Path,
        default=Path("data/analysis/retrieval_error_state.json"),
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest)
    _validate_manifest(manifest)
    dev_rows = manifest["dev"]
    eval_rows = manifest["eval"]
    mapping_build = build_mapping(dev_rows)
    all_mapping_norms = {norm for values in mapping_build.mapping.values() for norm in values}
    norm_ids = _load_norm_ids(args.norms)
    missing = all_mapping_norms - norm_ids
    if missing:
        raise ValueError(f"E-dev norms are absent from corpus: {sorted(missing)}")

    dev_logo = leave_one_group_out_predictions(dev_rows)
    dev_apparent = predict_rows(dev_rows, mapping_build.mapping)
    state: dict[str, Any] = {
        "version": 1,
        "seed": SEED,
        "oracle_issue_type": True,
        "manifest_version": manifest["manifest_version"],
        "edition_id": manifest["edition_id"],
        "counts": {"dev": len(dev_rows), "eval": len(eval_rows)},
        "mapping": _mapping_state(mapping_build),
        "dev": {
            "group_holdout": _evaluate(dev_rows, dev_logo),
            "apparent": _evaluate(dev_rows, dev_apparent),
        },
        "retriever_snapshot": _retriever_snapshot(
            Path("data/analysis/article_shortlist_fusion_state.json")
        ),
    }

    if args.split == "eval":
        overrides = _load_zero_shot(
            args.zero_shot,
            edition_id=str(manifest["edition_id"]),
            dev_rows=dev_rows,
            norm_ids=norm_ids,
        )
        zero_mapping = overlay_mapping(mapping_build.mapping, overrides)
        seen_types = set(mapping_build.mapping)
        seen_rows = [row for row in eval_rows if str(row["issue_type"]) in seen_types]
        state["zero_shot_overrides"] = overrides
        state["eval"] = {
            "issue_type_counts": dict(sorted(Counter(str(row["issue_type"]) for row in eval_rows).items())),
            "unseen_issue_types": sorted(
                {str(row["issue_type"]) for row in eval_rows} - seen_types
            ),
            "e_dev": _evaluate(eval_rows, predict_rows(eval_rows, mapping_build.mapping)),
            "seen_types": _evaluate(seen_rows, predict_rows(seen_rows, mapping_build.mapping)),
            "zero_shot": _evaluate(eval_rows, predict_rows(eval_rows, zero_mapping)),
        }

    report = _report(state)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _merge_aggregate(
        state,
        aggregate_state_path=args.aggregate_state,
        aggregate_report_path=args.aggregate_report,
        report_path=args.report,
    )
    print(report)
    print(f"wrote {args.report} and {args.state}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
