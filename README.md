# ZakupCheck

ZakupCheck — исследовательский прототип ассистента для проверки извещений о
государственных закупках на соответствие 44-ФЗ.

Публичный обезличенный benchmark доступен на Hugging Face:
[SergeySavinov/zakupcheck-44fz-retrieval-benchmark](https://huggingface.co/datasets/SergeySavinov/zakupcheck-44fz-retrieval-benchmark),
зафиксированная версия проекта — [`v1.0.0`](https://huggingface.co/datasets/SergeySavinov/zakupcheck-44fz-retrieval-benchmark/tree/v1.0.0).

Система должна принимать извещение и его приложения, находить потенциально
проблемные места и возвращать не просто вывод модели, а проверяемое замечание:

- дословный фрагмент документа;
- норму 44-ФЗ и дословную цитату из неё;
- тип возможного нарушения;
- вердикт `violation`, `uncertain` или `insufficient_evidence`.

ZakupCheck задуман как помощник эксперту: он подготавливает места для проверки,
но не заменяет юриста или ФАС и не выдаёт юридическое заключение.

## Для кого и для чего

Проект рассматривает два сценария работы с одним и тем же пайплайном:

1. **Самопроверка заказчика.** До публикации закупки проверить извещение и
   приложения, отдавая приоритет полноте найденных проблем.
2. **Проверка участником.** После публикации сопоставить конкретный довод или
   фрагмент документа с нормами закона, отдавая приоритет точности обоснования.

Основной исследовательский вопрос текущего этапа: как локально, без облачных
моделей, находить в полном тексте 44-ФЗ нужную статью и точную норму по языку
жалобы или закупочного документа.

## Как должен работать пайплайн

```text
извещение и приложения
        ↓
извлечение фрагментов и кандидатов нарушений
        ↓
поиск релевантных статей 44-ФЗ (BM25 + dense)
        ↓
cross-encoder reranking и выбор точных фрагментов нормы
        ↓
локальная LLM формирует замечание с дословными цитатами из документа и норм
        ↓
валидаторы проверяют существование нормы и дословность цитат
```

Закон хранится как дерево адресуемых норм — статья, часть, пункт. Поисковые
чанки используются только для retrieval и не подменяют юридическую ссылку на
норму. Редакция закона выбирается по дате публикации извещения.

## Текущий статус

Сейчас это **не готовое пользовательское приложение**, а воспроизводимый
исследовательский стенд. Завершён первый этап — корпус и сравнение retrieval:

- загрузка и структурный парсинг редакций 44-ФЗ;
- загрузка решений ФАС и извещений ЕИС;
- eval-набор по доводам жалоб с разметкой статей, точных норм и `issue_type`;
- BM25, dense retrieval, гибридный поиск и cross-encoder reranking;
- сравнение структурного и fixed-window чанкинга;
- анализ ошибок, доверительные интервалы и Baseline E
  (`issue_type → фиксированный список норм`);
- локальные CPU-бенчмарки моделей.

Генерация итоговых замечаний и сквозной пользовательский пайплайн ещё не
реализованы. Ближайший следующий этап — связать найденные нормы с фрагментами
извещения, добавить генерацию с обязательными цитатами и валидаторы
groundedness. Актуальные границы и план описаны в [DESIGN.md](DESIGN.md).

## Зафиксированная retrieval-конфигурация

Текущий основной вариант сначала выбирает статьи, затем отдельно собирает
точные доказательства:

- dense `Roflmax/bge-m3-russian-legal` на структурных `m700`-чанках;
- BM25 на структурных `m1500`-чанках;
- по 10 статей от каждого источника;
- cross-encoder `qilowoq/bge-reranker-v2-m3-en-ru`;
- three-way RRF по dense, BM25 и cross-encoder;
- отдельный evidence pack с несколькими точными фрагментами на статью.

На 52 `dev/blind`-кейсах article-stage достигает article R@10 `0.731`, а
evidence pack сохраняет эту полноту по статьям и повышает exact recall с
`0.202` до `0.314`. Сохранённый one-shot `eval/blind` snapshot дал article
R@10 `0.621` против `0.553` у dense baseline; 95% доверительный интервал
разницы — `[-0.045; +0.182]`, поэтому выигрыш на текущем eval ещё нельзя
считать доказанным.

Baseline E с известным вручную `issue_type` даёт на eval article/exact R@10
`0.727/0.606`. Это oracle upper bound, а не результат end-to-end системы:
в реальной работе тип нарушения ещё потребуется определить по документу.

Подробные настройки, вторичные метрики и ограничения вынесены из README в
[сводный анализ retriever](evals/reports/retrieval_error_analysis.md).

## Структура репозитория

```text
src/data_access/  загрузка и проверка публичного benchmark с Hugging Face
src/ingest/       загрузка и разбор закона, решений ФАС и извещений
src/schemas/      модели корпуса и eval-кейсов
src/index/        BM25, dense, fusion, reranking и метрики
src/validate/     проверка eval-разметки
scripts/          воспроизводимые загрузки и эксперименты
evals/            manifest, конфигурации и агрегированные отчёты
tests/            unit-тесты
data/             локальные документы, корпуса и кэши (не публикуются в git)
```

## Установка

Требуется Python 3.11 или новее.

```bash
python -m venv .venv
source .venv/bin/activate

# Базовый проект и тесты
pip install -e '.[dev]'

# Зависимости retrieval-моделей
pip install -e '.[retrieval]'

# Парсинг PDF/DOCX материалов ФАС
pip install -e '.[fas]'

# Загрузка публичного benchmark с Hugging Face
pip install -e '.[data]'
```

Для OCR сканов дополнительно нужны системные `tesseract-ocr` с русским
языковым пакетом и `poppler-utils`; для старых `.doc`/`.rtf` — LibreOffice.

## Минимальный рабочий сценарий

Скачать опубликованный benchmark `v1.0.0` и проверить его контрольные суммы:

```bash
python scripts/download_public_dataset.py
```

Данные появятся в
`data/hf/zakupcheck-44fz-retrieval-benchmark/v1.0.0/`. Путь можно изменить
через `--output-dir`, а другую версию выбрать через `--revision`. Загрузчик
проверяет `SHA256SUMS`, release manifest, число кейсов и отсутствие открытых
gold-меток в `eval`.

Публичный `dev` можно передавать существующим retrieval-скриптам вместо
закрытого каталога `evals/cases`:

```bash
python scripts/run_eval.py \
  --cases-dir data/hf/zakupcheck-44fz-retrieval-benchmark/v1.0.0 \
  --max-chunk 1500 \
  --query blind
```

Публичный `eval` загружается тем же loader, но его поля разметки равны `null`,
поэтому он предназначен для получения предсказаний, а не для локального
расчёта метрик. `load_verified()` автоматически исключает такие неразмеченные
кейсы. Полный список, включая blind eval, доступен через Python API:

```python
from pathlib import Path

from data_access.public_dataset import load_public_cases

cases = load_public_cases(
    Path("data/hf/zakupcheck-44fz-retrieval-benchmark/v1.0.0")
)
dev = [case for case in cases if case.split == "dev"]
eval_blind = [case for case in cases if case.split == "eval"]
```

HF-датасет содержит benchmark-кейсы, но не корпус закона. Для retrieval-прогона
редакцию 44-ФЗ по-прежнему нужно скачать и построить локально:

Скачать актуальную редакцию 44-ФЗ и построить структурный корпус:

```bash
python scripts/download_law.py
python scripts/build_law_corpus.py
```

Посмотреть доступные редакции или явно выбрать одну:

```bash
python scripts/download_law.py --list-redactions
python scripts/download_law.py --rdk 157
```

Проверить целостность eval-разметки и запустить тесты:

```bash
python scripts/validate_eval_cases.py
pytest -q --ignore=tests/test_download_fas.py
```

Тот же набор офлайн-тестов запускается в GitHub Actions на Python 3.11 и 3.12.
`test_download_fas.py` пока исключён: локально один из тестов обработки
HTML-карточки зависает, поэтому для него нужна отдельная диагностика.
Для локального запуска тестов достаточно `pip install -e '.[dev]'`;
скачивать модели и eval-данные не требуется.

Скачать отдельную жалобу или извещение можно по номеру:

```bash
python scripts/download_fas.py complaint --number 202600132489017625 --insecure
python scripts/download_fas.py notice --number 0373100062626000058 --insecure
```

Сырые документы сохраняются в `data/raw/`, публичный snapshot — в `data/hf/`,
построенные корпуса и модельные кэши — в других подкаталогах `data/`. Эта
директория не публикуется в git.

### Подготовка публичных eval-кейсов

Исходные кейсы не следует редактировать для публикации. Отдельные команды
строят локальный privacy-аудит и обезличенную копию-кандидат:

```bash
python scripts/audit_eval_privacy.py
python scripts/export_public_eval_cases.py
```

Автоматический проход закрывает структурированные идентификаторы и типичные
формы русских ФИО. Пропущенные сущности добавляются в отдельный JSON по образцу
`evals/public_redactions.example.json` и передаются обеим командам через
`--redactions`. Результат сохраняется в `data/public/`, не содержит локальных
путей и оффсетов и всегда помечается как требующий ручной покейсовой проверки.
По умолчанию gold-разметка `eval` не экспортируется; её раскрытие требует
явного флага `--include-eval-labels`.

После ручной покейсовой проверки кандидат собирается в отдельный неизменяемый
release-каталог. Финализатор требует выбранную лицензию данных, фиксирует
проверяющего, проверяет отсутствие eval-меток и локальных путей и создаёт
dataset card с контрольными суммами:

```bash
python scripts/finalize_public_dataset.py \
  --version 1.0.0 \
  --reviewed-by "privacy-reviewer" \
  --license-file path/to/reviewed-data-license.md
```

Готовый каталог появляется в
`data/public/releases/zakupcheck-benchmark-1.0.0/`. Существующий release
финализатор не перезаписывает.

## Воспроизведение исследований

Основные команды текущего этапа:

```bash
# Зафиксированный article-stage и evidence selection
python scripts/run_article_shortlist_fusion_eval.py --precision fp32 --focused
python scripts/run_m1000_evidence_eval.py

# Сравнение способов чанкинга
python scripts/run_chunking_comparison.py
python scripts/run_chunking_role_eval.py

# Baseline E: честный dev group-holdout и отдельный frozen eval
python scripts/run_baseline_e_eval.py
python scripts/run_baseline_e_eval.py --split eval
```

Модельные прогоны используют локальные кэши в `data/analysis/`. Eval-кейсы и
веса моделей в git не публикуются. `eval` используется только для однократной
отчётной проверки, а не для подбора конфигурации.

## Документация и отчёты

- [DESIGN.md](DESIGN.md) — постановка задачи, ограничения, архитектура и план;
- [сводный анализ retriever](evals/reports/retrieval_error_analysis.md) — результаты и причины ошибок;
- [article shortlist и BF16/FP32](evals/reports/article_shortlist_fusion.md);
- [сравнение `m700` и `m1000`](evals/reports/m1000_evidence_comparison.md);
- [сравнение способов чанкинга](evals/reports/chunking_ab_comparison.md);
- [роли article retrieval и evidence packing](evals/reports/chunking_role_comparison.md);
- [Baseline E](evals/reports/baseline_e.md);
- [runtime и квантование reranker](evals/reports/reranker_runtime_comparison.md);
- [аудит корпуса закона](evals/reports/law_corpus_audit.md);
- [CPU-бенчмарк локальной LLM](docs/Отчёт_llama-bench_T-lite.md).

## Данные и ограничения

В Git-репозитории публикуются код, небольшие конфигурации и агрегированные
результаты. Обезличенные benchmark-кейсы распространяются отдельно через
[Hugging Face](https://huggingface.co/datasets/SergeySavinov/zakupcheck-44fz-retrieval-benchmark).
Тексты 44-ФЗ, исходные документы ЕИС/ФАС, закрытые gold-метки `eval` и
модельные кэши в git и публичный датасет не входят.

Результаты экспериментов относятся к текущему небольшому eval-набору и не
доказывают юридическую корректность системы на произвольных закупках.
