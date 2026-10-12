# Article/chunk budget and two-pass reranking

Dense: `Roflmax/bge-m3-russian-legal`, m700; BM25 article source: m1500; evidence chunks: m700; cross-encoder: `qilowoq/bge-reranker-v2-m3-en-ru`.
Only dev/blind is used. Cross-encoder logits are reused from the existing cache; no model inference was performed.
A two-pass result is considered genuinely adaptive only when the shortlist removes at least 5 articles per query on average.
For every configuration the diagnostic path is identical: candidate article recall after dense/BM25 quotas; article recall after the optional first-pass shortlist; exact recall reachable anywhere in the final chunk pool; and article/exact recall in the returned top-10. Reported losses are differences between adjacent stages.

## Winners

### Joint surface

**D5+B10, 8 chunks/article**; mean/max scored chunks 96.1/112.
Candidate article recall 0.692 → shortlist 0.692 → final article R@10 0.692.
Exact pool ceiling 0.401 → final exact R@10 0.221; all-hit article@10 0.654.

### Two-pass

**D10+B10, first=2, shortlist=10, final=8**; mean/max scored chunks 89.1/98.
Candidate article recall 0.769 → shortlist 0.673 → final article R@10 0.673.
Exact pool ceiling 0.372 → final exact R@10 0.183; all-hit article@10 0.654.

Two-pass minus joint-surface article R@10: -0.019 (95% group-bootstrap [-0.130; +0.092]); gains 5, losses 5, ties 42.

## Best configuration under a hard per-query pair budget

| family | max pairs | configuration | mean pairs | exact R@10 | article R@10 | candidate R | shortlist R | pool exact ceiling |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| surface | 50 | D10+B10, 2 chunks/article | 31.1 | 0.163 | 0.673 | 0.769 | 0.769 | 0.266 |
| surface | 100 | D5+B10, 6 chunks/article | 72.8 | 0.173 | 0.673 | 0.692 | 0.692 | 0.333 |
| surface | 150 | D5+B10, 8 chunks/article | 96.1 | 0.221 | 0.692 | 0.692 | 0.692 | 0.401 |
| surface | 200 | D5+B10, 8 chunks/article | 96.1 | 0.221 | 0.692 | 0.692 | 0.692 | 0.401 |
| two_pass | 50 | D10+B10, first=1, shortlist=10, final=4 | 45.2 | 0.115 | 0.625 | 0.769 | 0.625 | 0.247 |
| two_pass | 100 | D10+B10, first=2, shortlist=10, final=8 | 89.1 | 0.183 | 0.673 | 0.769 | 0.673 | 0.372 |
| two_pass | 150 | D10+B10, first=2, shortlist=10, final=8 | 89.1 | 0.183 | 0.673 | 0.769 | 0.673 | 0.372 |
| two_pass | 200 | D10+B10, first=2, shortlist=10, final=8 | 89.1 | 0.183 | 0.673 | 0.769 | 0.673 | 0.372 |

## Pareto frontier

The frontier is non-dominated jointly on article R@10, exact R@10, and mean number of scored chunks.

| family | configuration | mean/max pairs | exact R@10 | article R@10 | candidate→shortlist→final article | pool→final exact |
|---|---|---:|---:|---:|---:|---:|
| surface | D5+B5, 1 chunks/article | 8.4/10 | 0.122 | 0.490 | 0.490→0.490→0.490 | 0.122→0.122 |
| surface | D10+B0, 1 chunks/article | 10.0/10 | 0.103 | 0.615 | 0.615→0.615→0.615 | 0.103→0.103 |
| surface | D0+B10, 1 chunks/article | 10.0/10 | 0.163 | 0.567 | 0.567→0.567→0.567 | 0.163→0.163 |
| surface | D5+B10, 1 chunks/article | 12.4/15 | 0.151 | 0.654 | 0.692→0.692→0.654 | 0.170→0.151 |
| surface | D0+B15, 1 chunks/article | 15.0/15 | 0.170 | 0.596 | 0.673→0.673→0.596 | 0.189→0.170 |
| surface | D0+B10, 2 chunks/article | 19.9/20 | 0.202 | 0.567 | 0.567→0.567→0.567 | 0.212→0.202 |
| surface | D5+B10, 2 chunks/article | 24.8/30 | 0.183 | 0.654 | 0.692→0.692→0.654 | 0.218→0.183 |
| surface | D10+B10, 2 chunks/article | 31.1/38 | 0.163 | 0.673 | 0.769→0.769→0.673 | 0.266→0.163 |
| surface | D5+B10, 6 chunks/article | 72.8/84 | 0.173 | 0.673 | 0.692→0.692→0.673 | 0.333→0.173 |
| two_pass | D0+B15, first=1, shortlist=10, final=8 | 83.4/85 | 0.202 | 0.596 | 0.673→0.596→0.596 | 0.381→0.202 |
| two_pass | D10+B10, first=2, shortlist=10, final=8 | 89.1/98 | 0.183 | 0.673 | 0.769→0.673→0.673 | 0.372→0.183 |
| two_pass | D5+B15, first=2, shortlist=10, final=8 | 91.6/97 | 0.202 | 0.635 | 0.750→0.635→0.635 | 0.353→0.202 |
| surface | D5+B10, 8 chunks/article | 96.1/112 | 0.221 | 0.692 | 0.692→0.692→0.692 | 0.401→0.221 |

Evaluated 305 surface and 1098 two-pass configurations (882 genuinely adaptive) over 52 dev cases. Full machine-readable results are stored locally in `data/analysis/article_chunk_budget_state.json`.
