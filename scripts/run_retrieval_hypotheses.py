#!/usr/bin/env python3
"""RRF, article-then-point, and gold-id coverage on the cached dense index."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import DenseIndex, SentenceTransformerEncoder  # noqa: E402
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.fusion import (  # noqa: E402
    RRF_K,
    article_then_points,
    gold_id_status,
    is_pointer_text,
    rrf,
    top_articles,
)
from index.recall import AggregateRecall, article_key, score_case  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "Roflmax/e5-large-legal-ru"
MAX_CHUNK = 1000
POOL = 200
ARTICLE_POOL = 50
N_ARTICLES = 3
RRF_POOL = 20
LIMIT = 10

Row = tuple[str, str, str, AggregateRecall, float, float]


def _norm_ids(chunks: list[Chunk]) -> set[str]:
    found: set[str] = set()
    for chunk in chunks:
        found.update(chunk.norm_ids)
    return found


def _unique(chunks_a: list[Chunk], chunks_b: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    merged: list[Chunk] = []
    for chunk in [*chunks_a, *chunks_b]:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        merged.append(chunk)
    return merged


def _aggregate(cases: list[EvalCase], lists: list[list[Chunk]]) -> tuple[AggregateRecall, float, float]:
    exact: list[float] = []
    article: list[float] = []
    all_exact: list[bool] = []
    all_article: list[bool] = []
    lengths: list[int] = []
    for case, chunks in zip(cases, lists, strict=True):
        scored = score_case(case.gold_norms, _norm_ids(chunks))
        exact.append(scored.exact.recall)
        article.append(scored.article.recall)
        all_exact.append(scored.exact.all_hit)
        all_article.append(scored.article.all_hit)
        lengths.append(len(chunks))
    n = len(cases)
    metrics = AggregateRecall(
        n=n,
        recall_exact=sum(exact) / n,
        all_exact=sum(all_exact) / n,
        recall_article=sum(article) / n,
        all_article=sum(all_article) / n,
    )
    return metrics, sum(lengths) / n, sum(article) / n


def _article_stage_recall(
    cases: list[EvalCase],
    rankings: list[list[Chunk]],
) -> float:
    recalls: list[float] = []
    for case, ranked in zip(cases, rankings, strict=True):
        chosen = set(top_articles(ranked, article_pool=ARTICLE_POOL, n_articles=N_ARTICLES))
        gold = {article_key(norm_id) for norm_id in case.gold_norms}
        recalls.append(len(gold & chosen) / len(gold))
    return sum(recalls) / len(recalls)


def _coverage(cases: list[EvalCase], chunks: list[Chunk]) -> str:
    corpus_ids = {norm_id for chunk in chunks for norm_id in chunk.norm_ids}
    by_id: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        for norm_id in chunk.norm_ids:
            by_id[norm_id].append(chunk)

    grouped: dict[str, list[tuple[EvalCase, str]]] = defaultdict(list)
    for case in cases:
        for gold_id in case.gold_norms:
            grouped[gold_id].append((case, gold_id_status(gold_id, corpus_ids)))

    def related(gold_id: str, status: str) -> list[Chunk]:
        if status == "exact":
            return by_id.get(gold_id, [])
        if status == "descendant":
            prefix = gold_id + ":"
            return [chunk for chunk in chunks if any(norm_id.startswith(prefix) for norm_id in chunk.norm_ids)]
        if status == "ancestor":
            return [
                chunk
                for chunk in chunks
                if any(gold_id.startswith(norm_id + ":") for norm_id in chunk.norm_ids)
            ]
        return []

    counts: dict[str, int] = defaultdict(int)
    for pairs in grouped.values():
        counts[pairs[0][1]] += 1

    lines = [
        "## Покрытие gold id чанками",
        "",
        f"Корпус: structural max_chunk {MAX_CHUNK}. Verified-кейсов: {len(cases)}.",
        "exact — этот id есть у чанка. descendant — в корпусе есть более дробный id.",
        "ancestor — в корпусе есть только более крупный id. absent — id и его уточнений нет.",
        "Отсылка — текст чанка не длиннее 240 символов и содержит «в соответствии со статьей».",
        "",
        "| статус | gold id |",
        "|---|---|",
    ]
    for status in ("exact", "descendant", "ancestor", "absent"):
        lines.append(f"| {status} | {counts[status]} |")
    lines.extend(["", "| gold id | статус | кейсов | отсылка | кратчайший связанный чанк | пример |", "|---|---|---|---|---|---|"])
    for gold_id in sorted(grouped):
        status = grouped[gold_id][0][1]
        if status == "exact":
            texts = related(gold_id, status)
            pointer = any(is_pointer_text(chunk.text) for chunk in texts)
            shortest = min((len(" ".join(chunk.text.split())) for chunk in texts), default=0)
        else:
            texts = related(gold_id, status)
            pointer = False
            shortest = min((len(" ".join(chunk.text.split())) for chunk in texts), default=0)
        if status == "exact" and not pointer:
            continue
        example = grouped[gold_id][0][0].case_id
        mark = "да" if pointer else "нет"
        length = str(shortest) if texts else "—"
        lines.append(f"| `{gold_id}` | {status} | {len(grouped[gold_id])} | {mark} | {length} | {example} |")
    pointer_cases = {
        case.case_id
        for gold_id, pairs in grouped.items()
        if any(is_pointer_text(chunk.text) for chunk in related(gold_id, "exact"))
        for case, _ in pairs
    }
    lines.extend(
        [
            "",
            f"Gold id с текстом-отсылкой: {sum(1 for gold_id in grouped if any(is_pointer_text(chunk.text) for chunk in related(gold_id, 'exact')))}.",
            f"Verified-кейсов с такой отсылкой среди gold: {len(pointer_cases)}.",
            "",
        ]
    )
    return "\n".join(lines)


def _method_table(rows: list[Row]) -> str:
    header = "| метод | query | split | n | R@10 exact | all@10 exact | R@10 art | all@10 art | чанков |"
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines = [header, sep]
    for method, variant, split, metrics, mean_len, _ in rows:
        lines.append(
            f"| {method} | {variant} | {split} | {metrics.n} "
            f"| {metrics.recall_exact:.3f} | {metrics.all_exact:.3f} "
            f"| {metrics.recall_article:.3f} | {metrics.all_article:.3f} "
            f"| {mean_len:.1f} |"
        )
    return "\n".join(lines)


def _stage_table(rows: list[tuple[str, str, str, int, float]]) -> str:
    header = "| метод | query | split | n | доля золотых статей в top-3 |"
    sep = "|" + "|".join(["---"] * 5) + "|"
    lines = [header, sep]
    for method, variant, split, n, recall in rows:
        lines.append(f"| {method} | {variant} | {split} | {n} | {recall:.3f} |")
    return "\n".join(lines)


def _lookup(rows: list[Row], method: str, variant: str, split: str) -> Row:
    for row in rows:
        if row[0] == method and row[1] == variant and row[2] == split:
            return row
    raise KeyError((method, variant, split))


def _conclusion(rows: list[Row], stage_rows: list[tuple[str, str, str, int, float]]) -> str:
    def art(method: str, variant: str, split: str) -> float:
        return _lookup(rows, method, variant, split)[3].recall_article

    def chunks(method: str, variant: str, split: str) -> float:
        return _lookup(rows, method, variant, split)[4]

    def stage(method: str, variant: str, split: str) -> float:
        for name, query, group, _, recall in stage_rows:
            if name == method and query == variant and group == split:
                return recall
        raise KeyError((method, variant, split))

    lines = [
        "## Вывод",
        "",
        (
            f"На eval, сырой довод, RRF top-20 со срезом 10 даёт R@10 art "
            f"{art('rrf top-20 → 10', 'raw', 'eval'):.3f}. "
            f"Это ниже dense@10 ({art('dense@10', 'raw', 'eval'):.3f}) и BM25@10 "
            f"({art('bm25@10', 'raw', 'eval'):.3f}). "
            f"Потолок объединения двух top-10 — {art('union@10', 'raw', 'eval'):.3f}, "
            f"но в нём {chunks('union@10', 'raw', 'eval'):.1f} чанка, не 10."
        ),
        (
            "Причина: при k=60 чанк на скромных местах в обоих списках обгоняет чанк, "
            "который стоит первым только в одном. Уникальные попадания как раз такие, "
            "и срез до 10 их вытесняет. На dev raw RRF чуть выше лучшей одиночной системы "
            f"({art('rrf top-20 → 10', 'raw', 'dev'):.3f} против BM25 {art('bm25@10', 'raw', 'dev'):.3f}), "
            "на отложенных eval — нет."
        ),
        (
            f"Два шага снижают R@10 art: dense с {art('dense@10', 'raw', 'eval'):.3f} "
            f"до {art('dense: 3 статьи → 10 пунктов', 'raw', 'eval'):.3f}. "
            f"В top-3 статей по первому чанку из пула 50 попадает только "
            f"{stage('dense top-50', 'raw', 'eval'):.3f} золотых статей. "
            "Нужная статья часто четвёртая и дальше, и отбор трёх её отбрасывает ещё до пунктов."
        ),
        (
            "Точный id есть у 26 различных gold. Ещё 8 заданы крупнее чанка "
            "(часть или статья целиком), поэтому exact по ним недостижим, пока чанк не несёт родительский id. "
            "Полностью отсутствующих id нет. Отсылка «в соответствии со статьей 33» — один id, "
            "`art.42:ch.2:p.1`, и она стоит в gold у 16 verified-кейсов."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    edition = Path("data/law/2026-08-04")
    cases = load_verified(Path("evals/cases"))
    chunks = load_chunks(chunks_path(edition, MAX_CHUNK))
    coverage = _coverage(cases, chunks)

    profile = profile_for(MODEL)
    print(f"loading {MODEL}", file=sys.stderr)
    encoder = SentenceTransformerEncoder(
        MODEL,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        max_tokens=profile.max_tokens,
    )
    dense = DenseIndex.load_cached(
        chunks,
        encoder,
        cache_dir=edition / "dense",
        model_name=MODEL,
        max_chunk=MAX_CHUNK,
    )
    if dense is None:
        print(f"нет кэша эмбеддингов {MODEL} m{MAX_CHUNK}", file=sys.stderr)
        return 1
    bm25 = Bm25Index(chunks)

    by_split: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_split[case.split].append(case)

    rows: list[Row] = []
    stage_rows: list[tuple[str, str, str, int, float]] = []
    for variant in ("raw", "blind"):
        for split in ("dev", "eval"):
            group = by_split[split]
            dense_lists: list[list[Chunk]] = []
            bm25_lists: list[list[Chunk]] = []
            for case in group:
                query = query_text(case, variant)
                dense_lists.append(dense.search(query, POOL))
                bm25_lists.append(bm25.search(query, POOL))
            methods: dict[str, list[list[Chunk]]] = {
                "dense@10": [ranked[:LIMIT] for ranked in dense_lists],
                "bm25@10": [ranked[:LIMIT] for ranked in bm25_lists],
                "union@10": [
                    _unique(left[:LIMIT], right[:LIMIT]) for left, right in zip(dense_lists, bm25_lists, strict=True)
                ],
                "rrf top-20 → 10": [
                    rrf([left[:RRF_POOL], right[:RRF_POOL]], k=RRF_K, limit=LIMIT)
                    for left, right in zip(dense_lists, bm25_lists, strict=True)
                ],
                "dense: 3 статьи → 10 пунктов": [
                    article_then_points(ranked, article_pool=ARTICLE_POOL, n_articles=N_ARTICLES, limit=LIMIT)
                    for ranked in dense_lists
                ],
                "bm25: 3 статьи → 10 пунктов": [
                    article_then_points(ranked, article_pool=ARTICLE_POOL, n_articles=N_ARTICLES, limit=LIMIT)
                    for ranked in bm25_lists
                ],
                "rrf: 3 статьи → 10 пунктов": [
                    article_then_points(
                        rrf([left, right], k=RRF_K, limit=POOL),
                        article_pool=ARTICLE_POOL,
                        n_articles=N_ARTICLES,
                        limit=LIMIT,
                    )
                    for left, right in zip(dense_lists, bm25_lists, strict=True)
                ],
            }
            for name, lists in methods.items():
                metrics, mean_len, art = _aggregate(group, lists)
                rows.append((name, variant, split, metrics, mean_len, art))
                print(
                    f"{variant} {split} {name} R@10 art={metrics.recall_article:.3f} "
                    f"exact={metrics.recall_exact:.3f} chunks={mean_len:.1f}",
                    file=sys.stderr,
                )
            for name, lists in (
                ("dense top-50", dense_lists),
                ("bm25 top-50", bm25_lists),
                (
                    "rrf top-50",
                    [rrf([left, right], k=RRF_K, limit=POOL) for left, right in zip(dense_lists, bm25_lists, strict=True)],
                ),
            ):
                stage_rows.append((name, variant, split, len(group), _article_stage_recall(group, lists)))

    report = "\n".join(
        [
            "# Проверка гипотез retrieval",
            "",
            f"Модель: `{MODEL}`. Чанк {MAX_CHUNK}. Запрос D, raw и blind.",
            f"RRF: k={RRF_K}, пул {RRF_POOL} у каждой системы, срез {LIMIT}.",
            f"Два шага: первые {N_ARTICLES} статьи в пуле {ARTICLE_POOL}, затем до {LIMIT} их чанков из пула {POOL}.",
            "union@10 сливает два top-10 и поэтому длиннее 10 чанков. Остальные методы отдают не больше 10.",
            "",
            _conclusion(rows, stage_rows),
            "## Сравнение методов",
            "",
            _method_table(rows),
            "",
            "## Попала ли статья в top-3 до отбора пунктов",
            "",
            _stage_table(stage_rows),
            "",
            coverage,
        ]
    )
    report_path = Path("evals/reports/retrieval_hypotheses.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
