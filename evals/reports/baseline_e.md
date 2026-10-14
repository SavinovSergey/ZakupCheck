# Baseline E: тип нарушения → нормы

Baseline E не ищет по тексту закона. Он получает gold `issue_type` и возвращает упорядоченный список norm id для конкретной редакции закона. Поэтому это **oracle upper bound для классификации типа**, а не end-to-end метрика продукта.

Основная оценка E-dev использует leave-one-`case_group_id`-out на dev. Затем full-dev таблица замораживается и применяется к eval. `other`, `unclear` и неизвестные типы намеренно не получают fallback.
Baseline E не читает текст довода, поэтому raw и blind для него тождественны.

## Агрегированные результаты

Основная метрика: article R@10. Exact и all-hit остаются вторичными. Интервал — 95% group bootstrap для article R@10, seed `20261008`.

| split | method | n | exact R@5 | article R@5 | exact R@10 | all exact@10 | article R@10 | all article@10 | article R@10 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | E-dev group holdout | 52 | 0.542 | 0.750 | 0.571 | 0.519 | 0.750 | 0.712 | [0.630; 0.857] |
| dev | E-dev apparent (diagnostic only) | 52 | 0.933 | 0.952 | 0.981 | 0.981 | 0.981 | 0.981 | [0.939; 1.000] |
| eval | E-dev frozen | 22 | 0.606 | 0.727 | 0.606 | 0.545 | 0.727 | 0.727 | [0.545; 0.909] |
| eval | E-dev frozen, dev-seen types | 19 | 0.702 | 0.842 | 0.702 | 0.632 | 0.842 | 0.842 | [0.684; 1.000] |
| eval | E-dev + zero-shot (post-hoc) | 22 | 0.742 | 0.864 | 0.742 | 0.682 | 0.864 | 0.864 | [0.727; 1.000] |

## Сравнение с замороженным retriever

Метрики retriever взяты из уже замороженного state; cross-encoder не перезапускался. Парный интервал не приводится, потому что этот state хранит агрегаты eval, а не все покейсовые предсказания.

| split | system | exact R@10 | article R@10 | all article@10 | precision |
|---|---|---:|---:|---:|---|
| dev | frozen hybrid + CE | 0.202 | 0.731 | 0.712 | fp32 |
| eval | frozen hybrid + CE snapshot | 0.152 | 0.621 | 0.591 | bf16 |

## Замороженная E-dev таблица

| issue_type | ordered norms (group support / case support) |
|---|---|
| `defective_object_description` | `44FZ:2026-08-04:art.33:ch.1:p.1` (2/2); `44FZ:2026-08-04:art.33:ch.1:p.2` (2/2); `44FZ:2026-08-04:art.42:ch.2:p.1` (2/2); `44FZ:2026-08-04:art.33:ch.1` (1/1); `44FZ:2026-08-04:art.33:ch.1:p.3` (1/1); `44FZ:2026-08-04:art.42:ch.1` (1/1); `44FZ:2026-08-04:art.42:ch.2` (1/1); `44FZ:2026-08-04:art.8:ch.2` (1/1) |
| `electronic_procedure_failure` | `44FZ:2026-08-04:art.24.1:ch.13` (2/4); `44FZ:2026-08-04:art.44:ch.5:p.3` (1/1) |
| `improper_participant_requirements` | `44FZ:2026-08-04:art.31:ch.6` (2/2); `44FZ:2026-08-04:art.31:ch.1:p.1` (1/1); `44FZ:2026-08-04:art.31:ch.5` (1/1); `44FZ:2026-08-04:art.42:ch.1:p.12` (1/1); `44FZ:2026-08-04:art.42:ch.2:p.4` (1/1) |
| `inconsistent_procurement_documents` | `44FZ:2026-08-04:art.42:ch.1:p.5` (1/1); `44FZ:2026-08-04:art.42:ch.5` (1/1); `44FZ:2026-08-04:art.7` (1/1) |
| `other_unlawful_rejection` | `44FZ:2026-08-04:art.48:ch.12:p.3` (1/1); `44FZ:2026-08-04:art.49:ch.5:p.1` (1/1) |
| `rejection_documents_or_qualification` | `44FZ:2026-08-04:art.31:ch.9` (1/1); `44FZ:2026-08-04:art.42:ch.2:p.1` (1/1); `44FZ:2026-08-04:art.48:ch.12:p.3` (1/1); `44FZ:2026-08-04:art.49:ch.5:p.1` (1/1); `44FZ:2026-08-04:art.50:ch.3:p.1` (1/1) |
| `rejection_product_or_information` | `44FZ:2026-08-04:art.49:ch.5:p.1` (7/7); `44FZ:2026-08-04:art.48:ch.12:p.8` (5/6); `44FZ:2026-08-04:art.49:ch.5` (2/2); `44FZ:2026-08-04:art.33:ch.1:p.1` (1/1); `44FZ:2026-08-04:art.48:ch.12:p.1` (1/1); `44FZ:2026-08-04:art.93:ch.12:p.6` (1/1) |
| `restrictive_object_description` | `44FZ:2026-08-04:art.42:ch.2:p.1` (7/7); `44FZ:2026-08-04:art.33:ch.1:p.1` (2/2); `44FZ:2026-08-04:art.33:ch.3` (1/1); `44FZ:2026-08-04:art.8` (1/1) |
| `statutory_preference_violation` | `44FZ:2026-08-04:art.51:ch.2:p.1` (1/1) |
| `unlawful_admission` | `44FZ:2026-08-04:art.48:ch.12:p.1` (1/1); `44FZ:2026-08-04:art.49:ch.5` (1/1); `44FZ:2026-08-04:art.49:ch.5:p.1` (1/1) |
| `other`, `unclear`, unknown | no fallback |

## Zero-shot sensitivity

Два issue type встречаются только в eval. Overlay ниже показан отдельно и не может называться слепым, так как eval manifest уже был просмотрен:

- `contract_conclusion_violation`: `44FZ:2026-08-04:art.51:ch.1`
- `unrelated_items_in_one_lot`: `44FZ:2026-08-04:art.42:ch.2:p.1`, `44FZ:2026-08-04:art.8`

## Dev group-holdout по issue type

Строки с n < 5 считаются только описательными.

| issue_type | n | exact R@10 | article R@10 | all article@10 | status |
|---|---:|---:|---:|---:|---|
| `defective_object_description` | 8 | 0.625 | 0.938 | 0.875 | comparison |
| `electronic_procedure_failure` | 5 | 0.800 | 0.800 | 0.800 | comparison |
| `improper_participant_requirements` | 4 | 0.375 | 1.000 | 1.000 | descriptive |
| `inconsistent_procurement_documents` | 3 | 0.000 | 0.667 | 0.667 | descriptive |
| `other_unlawful_rejection` | 2 | 0.000 | 0.000 | 0.000 | descriptive |
| `rejection_documents_or_qualification` | 4 | 0.000 | 0.000 | 0.000 | descriptive |
| `rejection_product_or_information` | 14 | 0.857 | 0.893 | 0.857 | comparison |
| `restrictive_object_description` | 8 | 0.896 | 0.938 | 0.875 | comparison |
| `statutory_preference_violation` | 1 | 0.000 | 0.000 | 0.000 | descriptive |
| `unclear` | 1 | 0.000 | 0.000 | 0.000 | descriptive |
| `unlawful_admission` | 2 | 0.000 | 0.750 | 0.500 | descriptive |

## Dev group-holdout по темам

| topic | n | exact R@10 | article R@10 | all article@10 | status |
|---|---:|---:|---:|---:|---|
| `application_review` | 27 | 0.593 | 0.667 | 0.630 | comparison |
| `competition` | 1 | 0.500 | 0.500 | 0.000 | descriptive |
| `contract_conclusion` | 1 | 0.000 | 0.000 | 0.000 | descriptive |
| `notice_content` | 6 | 0.667 | 1.000 | 1.000 | comparison |
| `object_description` | 12 | 0.639 | 0.875 | 0.833 | comparison |
| `participant_requirements` | 4 | 0.375 | 1.000 | 1.000 | descriptive |
| `single_supplier_procedure` | 1 | 0.000 | 0.000 | 0.000 | descriptive |

## Eval E-dev по issue type

| issue_type | n | exact R@10 | article R@10 | all article@10 | status |
|---|---:|---:|---:|---:|---|
| `contract_conclusion_violation` | 1 | 0.000 | 0.000 | 0.000 | descriptive |
| `defective_object_description` | 3 | 0.889 | 1.000 | 1.000 | descriptive |
| `improper_participant_requirements` | 1 | 1.000 | 1.000 | 1.000 | descriptive |
| `other_unlawful_rejection` | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| `rejection_documents_or_qualification` | 2 | 0.500 | 0.500 | 0.500 | descriptive |
| `rejection_product_or_information` | 5 | 0.600 | 0.600 | 0.600 | comparison |
| `restrictive_object_description` | 6 | 0.778 | 1.000 | 1.000 | comparison |
| `unlawful_admission` | 1 | 1.000 | 1.000 | 1.000 | descriptive |
| `unrelated_items_in_one_lot` | 2 | 0.000 | 0.000 | 0.000 | descriptive |

## Вывод

При oracle issue type защищённая от group leakage dev-оценка E-dev даёт article R@10 `0.750` и exact R@10 `0.571`. Это измеряет ценность таблицы после классификации, а не способность найти нарушение в документе.
Замороженная E-dev таблица даёт на eval article/exact R@10 `0.727/0.606`. Post-hoc zero-shot overlay изменяет их до `0.864/0.742`.
Retrieval остаётся нужен для неизвестных типов, выбора evidence и случаев, где upstream-детектор не уверен. Следующий end-to-end эксперимент должен заменить gold issue type на предсказание эвристики/LLM.

Machine-readable state: `data/analysis/baseline_e_state.json`.
