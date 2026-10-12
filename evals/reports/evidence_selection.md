# Evidence selection inside locked articles

Article shortlist is frozen to the dev winner: D10+B10, first=2, three-way article RRF k=30 with weights 1:1:2, top 10 articles. Eval is not used for tuning.
Candidate selectors compare source interleaving, chunk-level RRF, and diversity-aware RRF. Evidence rankers compare CE, chunk-level dense+BM25+CE RRF, and CE with retrieval bonus.

## Best 10-chunk result

**chunk_rrf:16; ce; one_per_article**: article R@10 0.731, exact R@10 0.215. Exact pool ceiling 0.529; mean/max scored candidates 153.2/160.
Against interleave:8 + CE + one/article, exact Δ +0.013 (95% group-bootstrap [-0.046; +0.073]).

## One chunk per article and expanded evidence packs

| output | configuration | output chunks | exact recall | all exact | article recall | pool exact ceiling |
|---|---|---:|---:|---:|---:|---:|
| one/article | chunk_rrf:16; ce; one_per_article | 10.0 | 0.215 | 0.173 | 0.731 | 0.529 |
| best top-10 | chunk_rrf:16; ce; one_per_article | 10.0 | 0.215 | 0.173 | 0.731 | 0.529 |
| exact-first top-10 | diverse_rrf:8@0.5; ce; top10_primary7_cap2 | 10.0 | 0.247 | 0.192 | 0.654 | 0.474 |
| pack-20 | interleave:8; ce; pack20_cap2 | 20.0 | 0.314 | 0.250 | 0.731 | 0.410 |
| pack-30 | diverse_rrf:16@0.25; ce; pack30_cap3 | 29.9 | 0.397 | 0.327 | 0.731 | 0.532 |

If top-10 is optimized for exact evidence instead of article recall, exact R@10 reaches 0.247 (Δ +0.045, 95% CI [-0.044; +0.133]), but article R@10 falls to 0.654.
Against the 10-chunk baseline, pack-20 exact recall Δ +0.112 (95% group-bootstrap [+0.043; +0.192]); pack-30 Δ +0.196 ([+0.102; +0.301]).
Chunk-level dense/BM25 bonuses did not beat plain cross-encoder ordering. Diversity is useful for a broad evidence pack, not for choosing the single best chunk per article.

Evaluated 480 configurations on 52 dev/blind cases. Full results are local in `data/analysis/evidence_selection_state.json`.
