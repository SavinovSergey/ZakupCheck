# Structural versus fixed-window retrieval

Scheme A uses legal hierarchy. Scheme B uses fixed windows over the whole edition text with overlap 200 characters; `norm_ids` are assigned by source-span intersection.
Important: this first table scores the union of norm ids in the first chunks. A window can cover several articles in one slot, so these article figures describe context-chunk coverage, not a clean top-10 article ranking. See `chunking_role_comparison.md` for the separated control.
Dense model: `Roflmax/bge-m3-russian-legal` m700. BM25: m1500 plus m2500 sensitivity. All methods return at most 10 article-diverse chunks. Hybrid parameters are selected on dev/blind only.

## Corpus

| scheme | m700 chunks | m1500 chunks | m2500 chunks | active articles covered |
|---|---:|---:|---:|---:|
| structural | 2079 | 1439 | 1237 | 84 |
| window | 1681 | 643 | 363 | 84 |

## Aggregate A/B matrix

Primary rows are blind queries. Raw is reported to expose citation leakage. The window hybrid is locked on dev/blind and then applied unchanged to eval and raw.

| split | query | method | structural exact/art/all-art R@5 | structural exact/art/all-art R@10 | window exact/art/all-art R@5 | window exact/art/all-art R@10 | window Δ article@10 (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|
| dev | blind | dense | 0.141/0.394/0.346 | 0.141/0.615/0.577 | 0.096/0.413/0.365 | 0.106/0.596/0.558 | -0.019 [-0.106; +0.060] |
| dev | blind | bm25_m1500 | 0.077/0.298/0.250 | 0.135/0.567/0.519 | 0.154/0.279/0.231 | 0.231/0.510/0.462 | -0.058 [-0.135; +0.010] |
| dev | blind | bm25_m2500 | 0.106/0.337/0.288 | 0.144/0.558/0.500 | 0.163/0.346/0.288 | 0.266/0.635/0.558 | +0.077 [-0.029; +0.192] |
| dev | blind | hybrid | 0.141/0.394/0.327 | 0.170/0.683/0.654 | 0.103/0.413/0.365 | 0.228/0.644/0.596 | -0.038 [-0.144; +0.069] |
| dev | raw | dense | 0.141/0.413/0.365 | 0.141/0.615/0.577 | 0.106/0.394/0.346 | 0.125/0.567/0.519 | -0.048 [-0.130; +0.019] |
| dev | raw | bm25_m1500 | 0.106/0.288/0.231 | 0.154/0.529/0.481 | 0.115/0.221/0.173 | 0.240/0.510/0.462 | -0.019 [-0.106; +0.060] |
| dev | raw | bm25_m2500 | 0.096/0.308/0.250 | 0.163/0.548/0.500 | 0.115/0.308/0.250 | 0.237/0.606/0.538 | +0.058 [-0.055; +0.170] |
| dev | raw | hybrid | 0.122/0.356/0.288 | 0.170/0.654/0.615 | 0.112/0.404/0.365 | 0.247/0.625/0.596 | -0.029 [-0.148; +0.091] |
| eval | blind | dense | 0.091/0.356/0.318 | 0.136/0.553/0.455 | 0.182/0.333/0.273 | 0.273/0.621/0.545 | +0.068 [-0.091; +0.227] |
| eval | blind | bm25_m1500 | 0.106/0.341/0.318 | 0.121/0.470/0.409 | 0.212/0.288/0.227 | 0.212/0.356/0.318 | -0.114 [-0.273; +0.023] |
| eval | blind | bm25_m2500 | 0.152/0.311/0.273 | 0.167/0.470/0.409 | 0.197/0.311/0.273 | 0.212/0.492/0.455 | +0.023 [-0.159; +0.205] |
| eval | blind | hybrid | 0.045/0.333/0.273 | 0.045/0.492/0.455 | 0.242/0.333/0.273 | 0.348/0.530/0.455 | +0.038 [-0.152; +0.220] |
| eval | raw | dense | 0.091/0.356/0.318 | 0.182/0.508/0.409 | 0.182/0.333/0.273 | 0.227/0.576/0.500 | +0.068 [-0.136; +0.273] |
| eval | raw | bm25_m1500 | 0.106/0.341/0.318 | 0.212/0.515/0.455 | 0.197/0.311/0.273 | 0.197/0.333/0.318 | -0.182 [-0.364; -0.023] |
| eval | raw | bm25_m2500 | 0.152/0.311/0.273 | 0.167/0.515/0.455 | 0.197/0.356/0.318 | 0.197/0.470/0.455 | -0.045 [-0.205; +0.114] |
| eval | raw | hybrid | 0.091/0.356/0.318 | 0.091/0.538/0.500 | 0.242/0.379/0.318 | 0.303/0.530/0.455 | -0.008 [-0.220; +0.189] |

## Locked hybrids

Structural dev winner: `BM25 m1500, k=30, w=1:2, articles=10`.
Window dev winner: `BM25 m2500, k=60, w=2:1, articles=10`.
The previous structural winner was also transferred unchanged to window chunks as a sensitivity check; it was not used to select the window winner.

| split | window hybrid | exact R@10 | article R@10 | all article@10 |
|---|---|---:|---:|---:|
| dev | transferred structural spec | 0.199 | 0.596 | 0.558 |
| dev | window dev winner | 0.228 | 0.644 | 0.596 |
| eval | transferred structural spec | 0.348 | 0.576 | 0.545 |
| eval | window dev winner | 0.348 | 0.530 | 0.455 |

Exact recall needs a granularity caveat: a fixed window carries every active article/part/point whose source span it intersects. A structural point chunk can carry only the point id. Part of the window exact gain therefore comes from hierarchical reachability, not solely better ranking; article R@10 remains the primary selection metric.

## Dev/blind topics

Topics with n < 5 are descriptive only.

| method | scheme | topic | n | exact R@10 | article R@10 | all article@10 | status |
|---|---|---|---:|---:|---:|---:|---|
| dense | structural | application_review | 27 | 0.074 | 0.389 | 0.333 | comparison |
| dense | structural | competition | 1 | 0.500 | 0.500 | 0.000 | descriptive |
| dense | structural | contract_conclusion | 1 | 1.000 | 1.000 | 1.000 | descriptive |
| dense | structural | notice_content | 6 | 0.000 | 0.833 | 0.833 | comparison |
| dense | structural | object_description | 12 | 0.236 | 0.833 | 0.833 | comparison |
| dense | structural | participant_requirements | 4 | 0.250 | 1.000 | 1.000 | descriptive |
| dense | structural | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| dense | window | application_review | 27 | 0.074 | 0.444 | 0.407 | comparison |
| dense | window | competition | 1 | 0.500 | 0.500 | 0.000 | descriptive |
| dense | window | contract_conclusion | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| dense | window | notice_content | 6 | 0.000 | 0.333 | 0.333 | comparison |
| dense | window | object_description | 12 | 0.125 | 0.875 | 0.833 | comparison |
| dense | window | participant_requirements | 4 | 0.375 | 1.000 | 1.000 | descriptive |
| dense | window | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| bm25_m1500 | structural | application_review | 27 | 0.185 | 0.667 | 0.630 | comparison |
| bm25_m1500 | structural | competition | 1 | 0.000 | 0.500 | 0.000 | descriptive |
| bm25_m1500 | structural | contract_conclusion | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| bm25_m1500 | structural | notice_content | 6 | 0.000 | 0.000 | 0.000 | comparison |
| bm25_m1500 | structural | object_description | 12 | 0.083 | 0.458 | 0.417 | comparison |
| bm25_m1500 | structural | participant_requirements | 4 | 0.250 | 0.875 | 0.750 | descriptive |
| bm25_m1500 | structural | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| bm25_m1500 | window | application_review | 27 | 0.222 | 0.574 | 0.556 | comparison |
| bm25_m1500 | window | competition | 1 | 0.000 | 0.500 | 0.000 | descriptive |
| bm25_m1500 | window | contract_conclusion | 1 | 1.000 | 1.000 | 1.000 | descriptive |
| bm25_m1500 | window | notice_content | 6 | 0.000 | 0.000 | 0.000 | comparison |
| bm25_m1500 | window | object_description | 12 | 0.292 | 0.500 | 0.417 | comparison |
| bm25_m1500 | window | participant_requirements | 4 | 0.375 | 0.625 | 0.500 | descriptive |
| bm25_m1500 | window | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| hybrid | structural | application_review | 27 | 0.130 | 0.648 | 0.630 | comparison |
| hybrid | structural | competition | 1 | 0.500 | 0.500 | 0.000 | descriptive |
| hybrid | structural | contract_conclusion | 1 | 1.000 | 1.000 | 1.000 | descriptive |
| hybrid | structural | notice_content | 6 | 0.000 | 0.667 | 0.667 | comparison |
| hybrid | structural | object_description | 12 | 0.236 | 0.667 | 0.667 | comparison |
| hybrid | structural | participant_requirements | 4 | 0.250 | 0.875 | 0.750 | descriptive |
| hybrid | structural | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| hybrid | window | application_review | 27 | 0.204 | 0.556 | 0.519 | comparison |
| hybrid | window | competition | 1 | 0.500 | 0.500 | 0.000 | descriptive |
| hybrid | window | contract_conclusion | 1 | 0.000 | 1.000 | 1.000 | descriptive |
| hybrid | window | notice_content | 6 | 0.000 | 0.667 | 0.667 | comparison |
| hybrid | window | object_description | 12 | 0.319 | 0.667 | 0.583 | comparison |
| hybrid | window | participant_requirements | 4 | 0.500 | 1.000 | 1.000 | descriptive |
| hybrid | window | single_supplier_procedure | 1 | 0.000 | 1.000 | 1.000 | descriptive |

## Conclusion

This first-pass chunk-union metric is not a clean article-ranking comparison because one fixed window can cover several articles in one result slot. On dev/blind, structural→window hybrid context coverage at ten chunks is 0.683→0.644 and exact R@10 is 0.170→0.228. On the one-shot eval/blind check, article coverage is 0.492→0.530 and exact R@10 is 0.045→0.348. All intervals include zero. Use the separated article-ranking and equal-budget evidence control in `chunking_role_comparison.md` for the actual decision.

Machine-readable state: `data/analysis/chunking_ab_comparison_state.json`. Eval was not used for parameter selection.
