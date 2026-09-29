| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |  99 |           pp512 |        254.92 ± 7.67 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |  99 |          pp2048 |       227.07 ± 11.02 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |  99 |           tg128 |         15.42 ± 0.38 |

build: 136887b66 (11221)
