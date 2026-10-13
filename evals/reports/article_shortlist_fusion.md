# Retrieval-aware article shortlist fusion

Dense: `Roflmax/bge-m3-russian-legal`, m700; BM25 article source: m1500; cross-encoder: `qilowoq/bge-reranker-v2-m3-en-ru` in FP32. Scope: `focused_locked_architecture`. Parameters were selected on dev/blind; eval is a single locked comparison when present.
Every row is decomposed into source candidate recall, shortlist recall, exact pool ceiling, and final top-10 recall. CE cost counts unique query/chunk pairs across both passes.

## Best result per shortlist strategy

| strategy | configuration | mean/max CE pairs | candidate→shortlist→final article | pool→final exact | all article@10 |
|---|---|---:|---:|---:|---:|
| ce_only | D10+B10; first=2; shortlist=10 | 89.1/98 | 0.769→0.673→0.673 | 0.372→0.183 | 0.654 |
| ce_retrieval_bonus | D10+B10; first=2; shortlist=10; alpha=0.1 | 89.4/98 | 0.769→0.673→0.673 | 0.372→0.183 | 0.654 |
| protected_quotas | D10+B10; first=2; shortlist=10; protect=5D+2B | 88.6/98 | 0.769→0.625→0.625 | 0.343→0.183 | 0.596 |
| three_way_rrf | D10+B10; first=2; shortlist=10; k=30; weights=1:1:2 | 89.9/98 | 0.769→0.731→0.731 | 0.410→0.202 | 0.712 |

## Overall winner

**three_way_rrf: D10+B10; first=2; shortlist=10; k=30; weights=1:1:2**.
Article R@10 0.731, exact R@10 0.202, all-hit article@10 0.712.
Against the best CE-only shortlist: Δ article R@10 +0.058 (95% group-bootstrap [+0.000; +0.132]); gains 3, losses 0, ties 49.
Against the previous D5+B10 × 8 joint-surface winner: Δ article R@10 +0.038 (95% group-bootstrap [-0.048; +0.132]). Exact R@10 changes from 0.221 to 0.202.

## Best result under a hard per-query CE budget

| max pairs | strategy | configuration | mean pairs | exact R@10 | article R@10 |
|---:|---|---|---:|---:|---:|
| 100 | three_way_rrf | D10+B10; first=2; shortlist=10; k=30; weights=1:1:2 | 89.9 | 0.202 | 0.731 |
| 150 | three_way_rrf | D10+B10; first=2; shortlist=10; k=30; weights=1:1:2 | 89.9 | 0.202 | 0.731 |
| 200 | three_way_rrf | D10+B10; first=2; shortlist=10; k=30; weights=1:1:2 | 89.9 | 0.202 | 0.731 |

Evaluated 22 configurations over 52 dev cases. Full machine-readable results are local in `data/analysis/article_shortlist_fusion_state.json`.

## FP32 versus BF16 on the locked architecture

Winner spec equal: `True`; identical ordered top-10 chunk rankings in 38/52 cases.
BF16→FP32 article R@10 0.731→0.731; exact R@10 0.202→0.202.
Measured throughput 6.44→2.52 pairs/s; estimated locked-pipeline CE latency 14.0→35.7 s/query.

## One-shot eval / blind (reporting snapshot)

This BF16 snapshot is preserved from the original locked run and was not rerun or used for FP32 tuning. Dense→RRF article R@10 0.553→0.621; exact R@10 0.136→0.152; all-hit article@10 0.455→0.591.
Δ article R@10 +0.068 (95% group-bootstrap [-0.045; +0.182]); gains 3, losses 1, ties 18.
