# ZakupCheck

Проверка извещений о закупке на соответствие 44-ФЗ с обязательным цитированием норм
(on-prem LLM, свой eval на решениях ФАС).

> Не юридическое заключение. Черновик мест для проверки человеком.

## Текущий статус retriever

Актуальная экспериментальная схема разделяет поиск статей и выбор точных
фрагментов:

1. Dense `Roflmax/bge-m3-russian-legal` на структурных `m700`-чанках и BM25
   на `m1500` независимо дают по 10 статей.
2. Для каждой статьи cross-encoder `qilowoq/bge-reranker-v2-m3-en-ru` оценивает два
   предварительных `m700`-чанка.
3. Ранги статей объединяются three-way RRF: `k=30`, веса dense:BM25:CE
   `1:1:2`. В article top-10 остаётся по одному представительному чанку на статью.
4. Для генератора строится отдельный evidence pack: до 8 кандидатов на статью,
   не более 2 финальных чанков на статью, порог CE-score `0.25`, hierarchy bonus
   `0.05` и оценочный бюджет 3000 токенов.

На 52 `dev/blind`-кейсах article-stage даёт article R@10 `0.731`, exact R@10 `0.202`
и all-hit article@10 `0.712`. Evidence pack сохраняет article R@10 `0.731` и поднимает
exact recall до `0.314` при 17.6 чанка и примерно 2692 условных токенах в среднем.
Сохранённый one-shot `eval/blind` snapshot (исходный BF16-run, reporting-only) дал article R@10
`0.621` и exact R@10 `0.152` против `0.553/0.136` у dense baseline; 95% CI article-разницы
`[-0.045; +0.182]`. Eval не перезапускался после FP32-контроля и не использовался для подбора.

### BF16 и FP32

Зафиксированная архитектура повторно проверена в FP32 без подмешивания BF16-scores.
Победитель и recall не изменились; порядок top-10 чанков полностью совпал в 38 из 52
кейсов. На текущем CPU BF16 даёт 6.44 пары/с против 2.52 пары/с у FP32;
расчётная CE-latency зафиксированного pipeline — 14.0 и 35.7 с/запрос соответственно.
Поэтому BF16 — предпочтительный CPU deployment-режим, а FP32 — контрольный эталон.

### m1000

`m1000` остаётся исследовательским вариантом. Его лучший dev exact recall равен `0.353`
против `0.314` у `m700`, но парная разница `+0.038` имеет 95% CI `[-0.040; +0.119]`.
Различие не доказано; FP32 evidence-reranking занимает 36.9 с/запрос против 31.3 с у `m700`.
Eval для этого подбора не использовался.

### Baseline E

Baseline E не ищет по тексту закона: он получает уже известный `issue_type` и
возвращает замороженный список норм. Честная leave-one-`case_group_id`-out оценка
на dev дала article/exact R@10 `0.750/0.571`. Замороженная full-dev таблица на eval дала
`0.727/0.606`, а на типах, встречавшихся в dev, article R@10 равен `0.842`.

Это oracle upper bound, потому что тип взят из ручной разметки. Post-hoc zero-shot overlay
для двух eval-only типов показал `0.864/0.742`, но не считается слепым eval. Для
end-to-end оценки ещё нужно заменить gold `issue_type` на выход эвристики или LLM.

## Документы

- [DESIGN.md](DESIGN.md) — дизайн (постановка, корпус, eval, план)
- [docs/Отчёт_llama-bench_T-lite.md](docs/Отчёт_llama-bench_T-lite.md) — замеры инференса
- [evals/reports/retrieval_error_analysis.md](evals/reports/retrieval_error_analysis.md) — сводный анализ retriever
- [evals/reports/article_shortlist_fusion.md](evals/reports/article_shortlist_fusion.md) — article shortlist и BF16/FP32
- [evals/reports/m1000_evidence_comparison.md](evals/reports/m1000_evidence_comparison.md) — `m700` против `m1000`
- [evals/reports/chunking_ab_comparison.md](evals/reports/chunking_ab_comparison.md) — structural против fixed-window
- [evals/reports/chunking_role_comparison.md](evals/reports/chunking_role_comparison.md) — article retrieval и evidence packing при равных бюджетах
- [evals/reports/baseline_e.md](evals/reports/baseline_e.md) — oracle `issue_type → нормы`
- [evals/reports/reranker_runtime_comparison.md](evals/reports/reranker_runtime_comparison.md) — runtime и квантование
- [evals/reports/law_corpus_audit.md](evals/reports/law_corpus_audit.md) — аудит корпуса

## Структура

См. §12 в DESIGN.md. Сырые данные и результаты парсинга — только в `data/` (не в git).

## Локальные зависимости вне репо

- **llama.cpp** — собирать отдельно (например `~/llama.cpp`), не клонировать сюда.
- **Модели GGUF** — например `~/models/t-lite/T-lite-it-2.1-Q5_K_M.gguf`.
- Инференс MVP: `llama-server` + Q5_K_M + Vulkan (`-ngl 99`).

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Скачать актуальную редакцию 44-ФЗ → data/law/{edition_id}/
python scripts/download_law.py
python scripts/download_law.py --list-redactions   # список rdk
# python scripts/download_law.py --rdk 157

# Жалоба / извещение ЕИС по номеру → data/raw/fas|notices/ (+ сырой текст; без разметки eval)
# python scripts/download_fas.py complaint --number 202600132489017625 --insecure
# python scripts/download_fas.py notice --number 0373100062626000058 --insecure
# Связать уже скачанные (reuse notice, без повторной загрузки):
# python scripts/download_fas.py link --complaint-dir data/raw/fas/202600132489017625 --link-only
# python scripts/download_fas.py complaint --number … --with-notice --link-only
# Пакетно (список номеров → жалобы + извещения + отчёт в evals/batch_reports/):
# python scripts/batch_download_fas.py --numbers-file evals/numbers_fas.txt --with-notice --insecure
# pip install 'zakup-check[fas]'   # PDF/CFF (pypdf + fontTools) + DOCX→Markdown (mammoth)
# OCR-fallback для сканов: системные tesseract-ocr (+ tessdata-rus) и poppler-utils
# .doc/.rtf: системный LibreOffice (soffice / libreoffice-writer)

# Проверить retrieval-разметку; ключ записи пересобирает замороженный manifest
python scripts/validate_eval_cases.py
python scripts/validate_eval_cases.py --write-manifest

# Воспроизвести FP32-проверку article-stage на dev
python scripts/run_article_shortlist_fusion_eval.py --precision fp32 --focused

# Повторить evidence-grid m700/m1000 на dev
python scripts/run_m1000_evidence_eval.py

# Собрать fixed-window corpus и сравнить BM25/dense/hybrid с structural
python scripts/run_chunking_comparison.py

# Разделить оценку article retrieval и evidence packing
python scripts/run_chunking_role_eval.py

# Baseline E: dev group-holdout (без eval) и явный frozen-eval
python scripts/run_baseline_e_eval.py
python scripts/run_baseline_e_eval.py --split eval

# Запустить unit-тесты без сетевого download-теста
pytest -q --ignore=tests/test_download_fas.py
```

Модельные прогоны используют локальные кэши в `data/analysis/`. Eval-кейсы и модели в git
не публикуются. Повторный запуск explicit `eval` не нужен для dev-подбора: сохранённые eval-
результаты считаются reporting-only.

Артефакты загрузки: `edition.json`, `full_text.txt`, `blocks.jsonl`, `raw/source.html`
(далее — `scripts/build_law_corpus.py` → NormUnit / Chunk).
Сырые данные: `data/raw/fas/{номер_жалобы}/`, `data/raw/notices/{regNumber}/`.

## Лицензия данных

В репозитории публикуется только код. Документы ЕИС/ФАС и дампы закона скачиваются локально скриптами.
