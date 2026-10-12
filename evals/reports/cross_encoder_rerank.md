# Cross-encoder reranking

Модель: `BAAI/bge-reranker-v2-m3`; CPU, max_length=512. Статьи выбираются консервативным dense+BM25 fusion, затем cross-encoder переранжирует m700-чанки внутри этих статей.

## Candidate ceiling, dev / blind

В пуле до 6 чанков на статью достижимы article recall 0.654 и exact recall 0.295. Это потолок пула, а не top-10.

## Dev / blind

Выбрано: `2` кандидата на статью, `10` обязательных статей. Dense → conservative hybrid → cross-encoder: article R@10 0.615→0.654→0.654; exact R@10 0.141→0.131→0.173.
Article-выигрыш целиком даёт candidate generation; cross-encoder меняет только выбор чанка внутри уже выбранной статьи.
Разница article R@10 +0.038 (95% group-bootstrap [-0.038; +0.120]); выигрышей 4, проигрышей 2, без изменений 46.

| candidates/article | mandatory articles | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 0.135 | 0.394 | 0.173 | 0.654 | 0.615 | 23489.5 |
| 6 | 10 | 0.125 | 0.394 | 0.144 | 0.654 | 0.615 | 23490.6 |
| 4 | 10 | 0.106 | 0.394 | 0.125 | 0.654 | 0.615 | 23490.2 |
| 2 | 8 | 0.135 | 0.394 | 0.151 | 0.529 | 0.481 | 23489.6 |
| 6 | 8 | 0.125 | 0.394 | 0.131 | 0.529 | 0.481 | 23490.6 |
| 4 | 8 | 0.106 | 0.394 | 0.112 | 0.529 | 0.481 | 23490.2 |
| 6 | 6 | 0.125 | 0.394 | 0.179 | 0.452 | 0.404 | 23490.5 |
| 4 | 6 | 0.106 | 0.394 | 0.160 | 0.452 | 0.404 | 23490.3 |
| 2 | 6 | 0.135 | 0.394 | 0.151 | 0.452 | 0.404 | 23489.6 |

\* latency включает retrieval, cross-encoder и постобработку; повторный запуск может читать scores из кэша, но в отчёте хранится измеренная первичная latency.

## Post-hoc validation eval / blind

Eval уже не является untouched test; результат только описательный.

| метод | exact R@10 | article R@10 | all article@10 |
|---|---:|---:|---:|
| dense m700 article-diverse | 0.136 | 0.553 | 0.455 |
| locked conservative hybrid | 0.152 | 0.530 | 0.455 |
| locked cross-encoder rerank | 0.152 | 0.530 | 0.455 |

Разница article R@10 -0.023 (95% CI [-0.136; +0.068]).
