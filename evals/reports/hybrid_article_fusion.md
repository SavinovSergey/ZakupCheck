# Article-level hybrid retrieval

Dense-компонент: `Roflmax/bge-m3-russian-legal`, structural `max_chunk=700`. BM25-компоненты: structural `max_chunk=1500` и `2500`. Основной запрос: blind.
Оба top-50 списка проецируются в ранги статей, объединяются weighted RRF, после чего итоговые evidence-чанки берутся только из dense m700.
Использован dense-кэш, обновлённый для текущей версии корпуса.

## Протокол выбора

На dev проверены `k ∈ {0,10,30,60}`, веса dense:BM25 `1:1`, `2:1`, `1:2`, BM25 m1500/m2500 и 5/8/10 статей при лимите 10 чанков. Победитель выбирается по article R@10; при разнице менее 0.01 — по exact R@10, all-hit article@10 и latency.

## Победитель dev / blind

**BM25 m1500, k=30, w=1:2, articles=10**: article R@10 0.683, exact R@10 0.170, all-hit article@10 0.654.
Против dense m700 article-diverse: Δ article R@10 +0.067 (95% group-bootstrap [-0.048; +0.184]); выигрышей 8, проигрышей 4, без изменений 40.

## Одиночные baselines, dev / blind

| метод | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |
|---|---:|---:|---:|---:|---:|---:|
| dense m700 article-diverse | 0.141 | 0.394 | 0.141 | 0.615 | 0.577 | 239.6 |
| BM25 m1500 article-diverse | 0.077 | 0.298 | 0.135 | 0.567 | 0.519 | 5.8 |
| BM25 m2500 article-diverse | 0.106 | 0.337 | 0.144 | 0.558 | 0.500 | 4.5 |

## Лучший гибрид для каждого BM25-размера

| конфигурация | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |
|---|---:|---:|---:|---:|---:|---:|
| BM25 m1500, k=30, w=1:2, articles=10 | 0.141 | 0.394 | 0.170 | 0.683 | 0.654 | 246.4 |
| BM25 m2500, k=60, w=2:1, articles=10 | 0.141 | 0.452 | 0.160 | 0.663 | 0.635 | 245.1 |

## Top-15 конфигураций dev / blind

| конфигурация | exact R@10 | article R@10 | all article@10 | ms/query* |
|---|---:|---:|---:|---:|
| BM25 m1500, k=30, w=1:2, articles=10 | 0.170 | 0.683 | 0.654 | 246.4 |
| BM25 m1500, k=10, w=1:2, articles=10 | 0.160 | 0.673 | 0.635 | 246.4 |
| BM25 m2500, k=60, w=2:1, articles=10 | 0.160 | 0.663 | 0.635 | 245.1 |
| BM25 m2500, k=30, w=1:1, articles=10 | 0.160 | 0.654 | 0.615 | 245.0 |
| BM25 m2500, k=30, w=2:1, articles=10 | 0.160 | 0.654 | 0.615 | 245.0 |
| BM25 m2500, k=60, w=1:1, articles=10 | 0.160 | 0.654 | 0.615 | 245.0 |
| BM25 m1500, k=30, w=1:1, articles=10 | 0.160 | 0.654 | 0.615 | 246.4 |
| BM25 m1500, k=60, w=1:1, articles=10 | 0.160 | 0.654 | 0.615 | 246.4 |
| BM25 m1500, k=60, w=1:2, articles=10 | 0.170 | 0.644 | 0.615 | 246.4 |
| BM25 m2500, k=0, w=2:1, articles=10 | 0.160 | 0.644 | 0.615 | 245.1 |
| BM25 m1500, k=10, w=2:1, articles=10 | 0.160 | 0.644 | 0.615 | 246.4 |
| BM25 m1500, k=10, w=1:1, articles=10 | 0.160 | 0.644 | 0.615 | 246.4 |
| BM25 m2500, k=60, w=1:2, articles=10 | 0.170 | 0.635 | 0.596 | 245.0 |
| BM25 m2500, k=30, w=1:2, articles=10 | 0.170 | 0.635 | 0.596 | 245.0 |
| BM25 m1500, k=0, w=2:1, articles=10 | 0.160 | 0.635 | 0.615 | 246.3 |

\* latency включает dense- и BM25-поиск плюс fusion; построение индексов не входит.
Полная сетка из 72 конфигураций сохранена локально в `data/analysis/hybrid_article_fusion_state.json`.

## Проверка зафиксированного варианта на eval / blind

| метод | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 |
|---|---:|---:|---:|---:|---:|
| dense m700 article-diverse | 0.091 | 0.356 | 0.136 | 0.553 | 0.455 |
| BM25 m1500 article-diverse | 0.106 | 0.341 | 0.121 | 0.470 | 0.409 |
| locked article-level hybrid | 0.045 | 0.333 | 0.045 | 0.492 | 0.455 |

Гибрид против dense article-diverse: Δ article R@10 -0.061 (95% group-bootstrap [-0.235; +0.098]); выигрышей 2, проигрышей 4, без изменений 16.
