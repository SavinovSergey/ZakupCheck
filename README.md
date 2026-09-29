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
```

Артефакты загрузки: `edition.json`, `full_text.txt`, `blocks.jsonl`, `raw/source.html`
(далее — `scripts/build_law_corpus.py` → NormUnit / Chunk).

## Лицензия данных

В репозитории публикуется только код. Документы ЕИС/ФАС и дампы закона скачиваются локально скриптами.
