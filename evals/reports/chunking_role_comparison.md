# Article retrieval and evidence packing by chunking scheme

This control separates two roles that the previous chunk-level R@10 mixed together. Article retrieval gives every article its own top-10 slot. Evidence packing keeps multi-article windows intact and compares equal context budgets.

## Article retrieval

A chunk contributes its rank to every intersected article, weighted by the fraction of chunk characters inside that article. Each article keeps only its best chunk contribution; this prevents a boundary sliver or a larger number of chunks from receiving a full extra rank.

| split | method | structural article/all-hit R@10 | window article/all-hit R@10 | window Δ article (95% CI) |
|---|---|---:|---:|---:|
| dev | dense | 0.615/0.577 | 0.567/0.519 | -0.048 [-0.127; +0.019] |
| dev | bm25_m1500 | 0.567/0.519 | 0.519/0.481 | -0.048 [-0.145; +0.047] |
| dev | bm25_m2500 | 0.558/0.500 | 0.615/0.538 | +0.058 [-0.056; +0.180] |
| dev | hybrid | 0.683/0.654 | 0.625/0.577 | -0.058 [-0.170; +0.055] |
| eval | dense | 0.553/0.455 | 0.621/0.545 | +0.068 [-0.091; +0.227] |
| eval | bm25_m1500 | 0.470/0.409 | 0.356/0.318 | -0.114 [-0.273; +0.023] |
| eval | bm25_m2500 | 0.470/0.409 | 0.356/0.318 | -0.114 [-0.318; +0.068] |
| eval | hybrid | 0.492/0.455 | 0.508/0.455 | +0.015 [-0.159; +0.182] |

Structural dev winner: `BM25 m1500, k=30, w=1:2`.  
Window dev winner: `BM25 m2500, k=30, w=1:1`.

### Projection sensitivity

First-seen projection gives every article touched by a chunk the same rank. The overlap-weighted projection is the primary control because it discounts small boundary intersections.

| split | scheme | method | first-seen article R@10 | overlap-weighted article R@10 |
|---|---|---|---:|---:|
| dev | structural | dense | 0.615 | 0.615 |
| dev | structural | bm25_m1500 | 0.567 | 0.567 |
| dev | structural | bm25_m2500 | 0.558 | 0.558 |
| dev | window | dense | 0.577 | 0.567 |
| dev | window | bm25_m1500 | 0.500 | 0.519 |
| dev | window | bm25_m2500 | 0.500 | 0.615 |
| eval | structural | dense | 0.553 | 0.553 |
| eval | structural | bm25_m1500 | 0.470 | 0.470 |
| eval | structural | bm25_m2500 | 0.470 | 0.470 |
| eval | window | dense | 0.530 | 0.621 |
| eval | window | bm25_m1500 | 0.356 | 0.356 |
| eval | window | bm25_m2500 | 0.492 | 0.356 |

## Evidence packing under equal budgets

The dev-selected article hybrid supplies ten articles. Dense m700 chunks are then packed with one representative per selected article first, followed by additional chunks in retrieval order. One token is approximated as three characters. `usable gold` requires at least 50% of the gold norm's source span, rather than merely carrying its `norm_id`.

| split | budget | scheme | label exact | usable gold | gold span coverage | article recall | chunks | chars | outside selected | extra articles | window Δ usable (95% CI) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 1500 | structural | 0.170 | 0.067 | 0.102 | 0.644 | 11.2 | 4487 | 0.000 | 0.00 | — |
| dev | 1500 | window | 0.122 | 0.048 | 0.066 | 0.471 | 6.0 | 4164 | 0.020 | 0.33 | -0.019 [-0.079; +0.031] |
| dev | 3000 | structural | 0.199 | 0.141 | 0.164 | 0.683 | 20.3 | 8988 | 0.000 | 0.00 | — |
| dev | 3000 | window | 0.218 | 0.131 | 0.143 | 0.635 | 12.4 | 8576 | 0.032 | 0.88 | -0.010 [-0.085; +0.061] |
| dev | 4500 | structural | 0.301 | 0.244 | 0.274 | 0.683 | 29.7 | 13488 | 0.000 | 0.00 | — |
| dev | 4500 | window | 0.237 | 0.179 | 0.177 | 0.644 | 19.0 | 13187 | 0.026 | 1.12 | -0.064 [-0.170; +0.042] |
| eval | 1500 | structural | 0.045 | 0.000 | 0.035 | 0.492 | 11.5 | 4487 | 0.000 | 0.00 | — |
| eval | 1500 | window | 0.152 | 0.045 | 0.052 | 0.311 | 6.0 | 4166 | 0.002 | 0.09 | +0.045 [+0.000; +0.136] |
| eval | 3000 | structural | 0.197 | 0.182 | 0.161 | 0.492 | 21.2 | 8982 | 0.000 | 0.00 | — |
| eval | 3000 | window | 0.318 | 0.152 | 0.144 | 0.508 | 12.4 | 8575 | 0.029 | 0.77 | -0.030 [-0.197; +0.121] |
| eval | 4500 | structural | 0.212 | 0.212 | 0.191 | 0.492 | 31.0 | 13484 | 0.000 | 0.00 | — |
| eval | 4500 | window | 0.318 | 0.197 | 0.175 | 0.508 | 19.0 | 13187 | 0.020 | 0.77 | -0.015 [-0.212; +0.167] |

### Conditional evidence control

For this diagnostic only, both packers receive the gold articles. It isolates evidence selection from article-retrieval errors; it is not a deployable score.

| split | budget | scheme | usable gold | gold span coverage | outside gold articles |
|---|---:|---|---:|---:|---:|
| dev | 1500 | structural | 0.603 | 0.579 | 0.000 |
| dev | 1500 | window | 0.458 | 0.412 | 0.054 |
| dev | 3000 | structural | 0.811 | 0.782 | 0.000 |
| dev | 3000 | window | 0.679 | 0.638 | 0.067 |
| dev | 4500 | structural | 0.904 | 0.880 | 0.000 |
| dev | 4500 | window | 0.849 | 0.805 | 0.060 |
| eval | 1500 | structural | 0.629 | 0.637 | 0.000 |
| eval | 1500 | window | 0.424 | 0.453 | 0.071 |
| eval | 3000 | structural | 0.765 | 0.767 | 0.000 |
| eval | 3000 | window | 0.697 | 0.685 | 0.082 |
| eval | 4500 | structural | 0.894 | 0.894 | 0.000 |
| eval | 4500 | window | 0.780 | 0.779 | 0.073 |

## Conclusion

After separating roles, the comparison no longer treats several articles in one window as several free top-10 hits. Dev/blind hybrid article R@10 is 0.683 for structural and 0.625 for window. Under a 3000-token evidence budget, usable gold recall is 0.141 versus 0.131; window outside-selected text is 0.032. The two numbers answer different questions: article ranking chooses legal provisions, while evidence packing measures the context actually delivered to generation.
