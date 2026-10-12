# Retrieval-aware article shortlist fusion

Dense: `Roflmax/bge-m3-russian-legal`, m700; BM25 article source: m1500; cross-encoder: `qilowoq/bge-reranker-v2-m3-en-ru`. Parameters were selected on dev/blind; eval is a single locked comparison when present.
Every row is decomposed into source candidate recall, shortlist recall, exact pool ceiling, and final top-10 recall. CE cost counts unique query/chunk pairs across both passes.

## Best result per shortlist strategy

| strategy | configuration | mean/max CE pairs | candidate→shortlist→final article | pool→final exact | all article@10 |
|---|---|---:|---:|---:|---:|
| ce_only | D10+B10; first=2; shortlist=10 | 89.1/98 | 0.769→0.673→0.673 | 0.372→0.183 | 0.654 |
| ce_retrieval_bonus | D0+B50; first=2; shortlist=10; alpha=0.1 | 157.8/160 | 0.962→0.702→0.702 | 0.362→0.183 | 0.673 |
| protected_quotas | D10+B10; first=1; shortlist=10; protect=5D+2B | 83.0/89 | 0.769→0.663→0.663 | 0.381→0.202 | 0.635 |
| retrieval_rrf | D10+B10; shortlist=10; k=60; weights=1:1 | 78.9/80 | 0.769→0.683→0.683 | 0.401→0.212 | 0.654 |
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

Evaluated 1113 configurations over 52 dev cases. Full machine-readable results are local in `data/analysis/article_shortlist_fusion_state.json`.

## One-shot eval / blind

Dense m700 article-diverse: article R@10 0.553, exact R@10 0.136, all-hit article@10 0.455.
Locked three-way RRF: article R@10 0.621, exact R@10 0.152, all-hit article@10 0.591.
Δ article R@10 +0.068 (95% group-bootstrap [-0.045; +0.182]); gains 3, losses 1, ties 18. Mean/max required CE pairs 91.3/100 per query.
