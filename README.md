# ZakupCheck

Проверка извещений о закупке на соответствие 44-ФЗ с обязательным цитированием норм
(on-prem LLM, свой eval на решениях ФАС).

> Не юридическое заключение. Черновик мест для проверки человеком.

## Документы

- [DESIGN.md](DESIGN.md) — дизайн (постановка, корпус, eval, план)
- [Проект_44ФЗ_план.md](Проект_44ФЗ_план.md) — краткий план
- [docs/Отчёт_llama-bench_T-lite.md](docs/Отчёт_llama-bench_T-lite.md) — замеры инференса

## Структура

См. §12 в DESIGN.md. Сырые данные и результаты парсинга — только в `data/` (не в git).

## Локальные зависимости вне репо

- **llama.cpp** — собирать отдельно (например `~/llama.cpp`), не клонировать сюда.
- **Модели GGUF** — например `~/models/t-lite/T-lite-it-2.1-Q5_K_M.gguf`.
- Инференс MVP: `llama-server` + Q5_K_M + Vulkan (`-ngl 99`).

## Быстрый старт (позже)

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
# pip install 'zakup-check[fas]'   # PDF (pypdf) + DOCX→Markdown (mammoth)
# OCR-fallback для сканов: системные tesseract-ocr (+ tessdata-rus) и poppler-utils
# .doc/.rtf: системный LibreOffice (soffice / libreoffice-writer)
```

Артефакты загрузки: `edition.json`, `full_text.txt`, `blocks.jsonl`, `raw/source.html`
(далее — `scripts/build_law_corpus.py` → NormUnit / Chunk).
Сырые данные: `data/raw/fas/{номер_жалобы}/`, `data/raw/notices/{regNumber}/`.

## Лицензия данных

В репозитории публикуется только код. Документы ЕИС/ФАС и дампы закона скачиваются локально скриптами.
