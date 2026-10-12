# Conservative hybrid and within-article reranking

Dense m700 сохраняет верхнюю квоту статей; BM25 m1500/m2500 заполняет только оставшиеся слоты. Внутри выбранных статей чанки общего m700-корпуса объединяются chunk-level RRF dense+BM25.
Основной запрос — blind; лимит — 10 финальных чанков. Использован dense-кэш текущей версии корпуса.

## Dev / blind

Выбран **BM25 m1500, keep=7, article w=1:1, articles=10, chunk w=1:2**: article R@10 0.615→0.654, exact R@10 0.141→0.131.
Разница article R@10 +0.038 (95% group-bootstrap [-0.038; +0.120]); выигрышей 4, проигрышей 2, без изменений 46.

## Top-15 конфигураций

| конфигурация | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |
|---|---:|---:|---:|---:|---:|---:|
| BM25 m1500, keep=8, article w=1:1, articles=10, chunk w=1:2 | 0.074 | 0.394 | 0.131 | 0.654 | 0.615 | 264.3 |
| BM25 m1500, keep=7, article w=1:1, articles=10, chunk w=1:2 | 0.074 | 0.394 | 0.131 | 0.654 | 0.615 | 264.3 |
| BM25 m2500, keep=9, article w=1:1, articles=10, chunk w=1:2 | 0.074 | 0.394 | 0.112 | 0.654 | 0.615 | 262.6 |
| BM25 m2500, keep=8, article w=2:1, articles=10, chunk w=1:2 | 0.074 | 0.394 | 0.112 | 0.654 | 0.615 | 262.6 |
| BM25 m1500, keep=9, article w=1:1, articles=10, chunk w=1:2 | 0.074 | 0.394 | 0.112 | 0.654 | 0.615 | 264.3 |
| BM25 m2500, keep=9, article w=1:1, articles=10, chunk w=1:1 | 0.045 | 0.394 | 0.083 | 0.654 | 0.615 | 262.6 |
| BM25 m1500, keep=9, article w=1:1, articles=10, chunk w=1:1 | 0.045 | 0.394 | 0.083 | 0.654 | 0.615 | 264.3 |
| BM25 m1500, keep=7, article w=1:1, articles=10, chunk w=1:1 | 0.045 | 0.394 | 0.083 | 0.654 | 0.615 | 264.3 |
| BM25 m1500, keep=8, article w=1:1, articles=10, chunk w=1:1 | 0.045 | 0.394 | 0.083 | 0.654 | 0.615 | 264.3 |
| BM25 m2500, keep=9, article w=1:1, articles=10, chunk w=2:1 | 0.054 | 0.394 | 0.074 | 0.654 | 0.615 | 262.6 |
| BM25 m1500, keep=8, article w=1:1, articles=10, chunk w=2:1 | 0.054 | 0.394 | 0.074 | 0.654 | 0.615 | 264.3 |
| BM25 m1500, keep=9, article w=1:1, articles=10, chunk w=2:1 | 0.054 | 0.394 | 0.074 | 0.654 | 0.615 | 264.3 |
| BM25 m1500, keep=7, article w=1:1, articles=10, chunk w=2:1 | 0.054 | 0.394 | 0.074 | 0.654 | 0.615 | 264.3 |
| BM25 m2500, keep=8, article w=2:1, articles=10, chunk w=1:1 | 0.045 | 0.394 | 0.064 | 0.654 | 0.615 | 262.6 |
| BM25 m2500, keep=8, article w=2:1, articles=10, chunk w=2:1 | 0.054 | 0.394 | 0.054 | 0.654 | 0.615 | 262.6 |

\* latency включает dense m700, BM25 для статей, BM25 m700 и обе стадии fusion; индексация не входит.
Полная сетка сохранена локально в `data/analysis/conservative_hybrid_state.json`.

## Post-hoc validation на eval / blind

Этот набор уже использовался при анализе первого гибрида, поэтому результат не является новым untouched test.

| метод | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 |
|---|---:|---:|---:|---:|---:|
| dense m700 article-diverse | 0.091 | 0.356 | 0.136 | 0.553 | 0.455 |
| locked conservative hybrid | 0.106 | 0.356 | 0.152 | 0.530 | 0.455 |

Разница с dense article-diverse по article R@10: -0.023 (95% CI [-0.136; +0.068]); выигрышей 1, проигрышей 1, без изменений 20.
