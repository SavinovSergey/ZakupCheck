# Сравнение runtimes для bge-reranker-v2-m3

INT8 и GGUF измерены на 52 dev/blind кейсах и 3084 одинаковых парах «запрос—m700-чанк». Bilingual PyTorch и ONNX FP32 проверены на 10 кейсах как контроль эквивалентности; PyTorch FP32 — эталонные scores из первичного прогона. Latency измерена свежим запуском на первых 10 кейсах, 8 физических CPU-ядер, max_length/context 512.

Текущий пул содержит до 10 статей на запрос. Поэтому чувствительная article-метрика здесь — R@5 и overlap top-5; article R@10 почти фиксирован candidate generation.

## Итог

| runtime | quality n | model MB | peak RSS MB | load s | p50 ms/query | p95 ms/query | pairs/s | speedup | Spearman | chunk top-10 overlap | article top-5 overlap | exact R@10 | article R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch-fp32 | 52 | 2271 | 3186 | 4.37 | 23104 | 31425 | 2.6 | 1.00x | 1.0000 | 1.000 | 1.000 | 0.144 | 0.481 |
| pytorch-fp32-en-ru | 10 | 1500 | 3096 | 3.38 | 21932 | 31595 | 2.6 | 1.02x | 0.9999 | 1.000 | 1.000 | 0.000 | 0.250 |
| onnx-fp32 | 10 | 1500 | 5590 | 1.48 | 30492 | 41259 | 1.9 | 0.73x | 0.9999 | 1.000 | 1.000 | 0.000 | 0.250 |
| onnx-int8 | 52 | 376 | 4447 | 1.66 | 13089 | 18573 | 4.5 | 1.75x | 0.9697 | 0.865 | 0.896 | 0.135 | 0.452 |
| gguf-q8 | 52 | 636 | 1281 | 1.78 | 13025 | 20392 | 4.3 | 1.68x | 0.9467 | 0.854 | 0.877 | 0.144 | 0.452 |

## Fidelity к PyTorch FP32

- `pytorch-fp32-en-ru`: chunk top-1 1.000, article top-1 1.000; Δ article R@5 +0.000, 95% CI [+0.000; +0.000].
- `onnx-fp32`: chunk top-1 1.000, article top-1 1.000; Δ article R@5 +0.000, 95% CI [+0.000; +0.000].
- `onnx-int8`: chunk top-1 0.731, article top-1 0.769; Δ article R@5 -0.029, 95% CI [-0.089; +0.020].
- `gguf-q8`: chunk top-1 0.731, article top-1 0.750; Δ article R@5 -0.029, 95% CI [-0.075; +0.000].

## Прямое сравнение PyTorch FP32

Truncated en-ru: speedup 1.02x, парный bootstrap 95% CI [0.996x; 1.044x], быстрее в 6 из 10 кейсов. model 1500 MB против 2271 MB, peak RSS 3096 MB против 3186 MB. Fidelity на 10 одинаковых кейсах: Spearman 0.9999, chunk top-10 overlap 1.000, article top-5 overlap 1.000; MAE logits 0.000164, max abs 0.055319.

Tokenizer en-ru полностью совпал с оригинальным на 97.99% пар; `<unk>`: 0.000000.

## Tokenizer parity en-ru

Доля пар с полностью совпадающей последовательностью токенов: 0.9799; `<unk>`: 0.000000.

## Решение

Критерий допуска: Δ article R@5 не хуже -0.02, chunk top-10 overlap ≥0.95, article top-5 overlap ≥0.95 и измеримое ускорение. Победитель определяется по этим ограничениям, затем по latency и памяти.

**Ни ONNX INT8, ни GGUF Q8_0 критерий fidelity не прошли.** Их актуальные speedup, overlap и retrieval-метрики приведены в таблице выше; пересчёт speedup использует тот же свежий PyTorch baseline.

Bilingual PyTorch FP32 уменьшает файл весов примерно на треть и быстрее загружается, но ускорение самого reranking на этом прогоне статистически неотличимо от шума. Для длинного batch-прогона переход не сокращает расчётное время существенно; его смысл — меньший артефакт и более быстрый cold start. Перед production-заменой нужен полный 52-case fidelity-прогон. Bilingual ONNX FP32 на контрольных 10 кейсах практически эквивалентен оригиналу (Spearman 0.9999, top-10 overlap 1.000), но в ONNX Runtime CPU оказался медленнее PyTorch. Поэтому для следующего широкого эксперимента пока оставляем оригинальный PyTorch FP32; если время расчёта критично, INT8 можно использовать только как явный quality–speed trade-off, а не как эквивалентную замену.

GGUF-прогон одновременно меняет квантование и runtime/tokenization, поэтому расхождение нельзя приписать только Q8_0 без FP16/FP32 GGUF-контроля.

Полные scores и профили хранятся локально в `data/analysis/reranker_runtime/`; этот каталог gitignored.
