# m700 evidence refinement

> **Статус:** соседи, structured-prefix и hierarchy-сетка ниже считались в BF16 и остаются
> историческими. Победившая рабочая сетка пересчитана в FP32: конфигурация и exact recall
> `0.314` не изменились. Актуальный контроль: `m1000_evidence_comparison.md`.

Dev/blind only. Article shortlist is frozen to D10+B10, first-pass=2, three-way article RRF 1:1:2, top 10. Eval is not used.

## Results

| method | chunks | estimated tokens | exact recall | all exact | article recall | pool exact |
|---|---:|---:|---:|---:|---:|---:|
| fixed pack-20 | 20.0 | 3139 | 0.314 | 0.250 | 0.731 | 0.410 |
| base_plus_neighbors radius 2 | 20.0 | 3054 | 0.311 | 0.250 | 0.731 | 0.529 |
| plain CE pack-20 (4 candidates/article) | 20.0 | 3257 | 0.266 | 0.212 | 0.731 | 0.304 |
| structured CE pack-20 (4 candidates/article) | 20.0 | 3387 | 0.247 | 0.192 | 0.731 | 0.304 |
| hierarchy base8 bonus 0.1 | 20.0 | 3099 | 0.314 | 0.250 | 0.731 | 0.410 |
| dynamic base8 t3000 r0.25 h0.05 | 17.6 | 2684 | 0.314 | 0.250 | 0.731 | 0.410 |

Neighbour expansion vs pack-20: exact Δ -0.003 (95% CI [-0.063; +0.057]).
Structured CE input vs the identical four-candidate plain-text pool: exact Δ -0.019 (95% CI [-0.061; +0.000]).
Best dynamic pack vs pack-20: exact Δ +0.000 (95% CI [+0.000; +0.000]).

The selected dynamic rule keeps the same per-case exact recall while reducing mean output from 20.0 to 17.6 chunks and estimated context from 3139 to 2684 tokens. It uses the original eight-candidate pool, so it does not increase cross-encoder work.
Neighbour-only evidence and structured prefixes are rejected: neither improves exact recall. Hierarchy bonus changes which chunks are returned but does not improve recall.

Estimated tokens use ceil(characters / 3), because the generator tokenizer is not fixed yet.

## Oracle decomposition of missed gold norms

`{"hit": 20, "article_not_selected": 15, "exact_unavailable_in_m700": 8, "candidate_miss": 9, "selection_miss": 14}`

Full local grid: `data/analysis/m700_evidence_refinement_state.json` (133 configurations).
