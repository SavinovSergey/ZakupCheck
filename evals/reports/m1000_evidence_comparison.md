# m1000 evidence comparison

Dev/blind only, cross-encoder FP32. The article shortlist is frozen to the previously locked m700/BM25 article selector; only evidence chunking and within-article candidates change.

## Controlled comparison

| method | candidates/article | output chunks | estimated tokens | exact recall | all exact | article recall | pool exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| m700 fixed20 c8 | 8 | 20.0 | 3136 | 0.314 | 0.250 | 0.731 | 0.410 |
| m700 locked dynamic c8 | 8 | 17.6 | 2692 | 0.314 | 0.250 | 0.731 | 0.410 |
| m700 dynamic c8 t3000 r0.25 h0.05 | 8 | 17.6 | 2692 | 0.314 | 0.250 | 0.731 | 0.410 |
| m1000 fixed20 c8 | 8 | 20.0 | 3725 | 0.343 | 0.288 | 0.731 | 0.449 |
| m1000 locked dynamic c8 | 8 | 16.2 | 2822 | 0.304 | 0.250 | 0.731 | 0.449 |
| m1000 dynamic c8 t2500 r0 h0.05 | 8 | 15.0 | 2482 | 0.353 | 0.308 | 0.731 | 0.449 |
| m1000 pack30 c8 | 8 | 29.8 | 5639 | 0.391 | 0.346 | 0.731 | 0.449 |

Fixed-20 m1000 versus fixed-20 m700: exact Δ +0.029 (95% group-bootstrap [-0.049; +0.109]).
Locked dynamic m1000 versus locked dynamic m700: exact Δ -0.010 (95% group-bootstrap [-0.090; +0.066]).
Exploratory best m1000 versus exploratory best m700: exact Δ +0.038 (95% group-bootstrap [-0.040; +0.119]).

The same dev sensitivity grid is rerun for m700 and m1000: 4/8 candidates per article, fixed pack-20, fixed pack-30, and dynamic score/token thresholds. Selection remains dev-only.
Both corpora were scored with `qilowoq/bge-reranker-v2-m3-en-ru` in FP32. No score was imported from a BF16 cache. Reused original-model FP32 scores: 0.
Measured CE inference: m700 31.3 s/query (2.52 pairs/s), m1000 36.9 s/query (2.13 pairs/s). m1000 is 1.18× slower per query for these pools.

## Oracle decomposition for locked m1000

`{"hit": 19, "article_not_selected": 15, "candidate_miss": 15, "selection_miss": 9, "exact_unavailable_in_m1000": 8}`

Full local grid: `data/analysis/m1000_evidence_state.json` (78 m700 and 78 m1000 configurations).
