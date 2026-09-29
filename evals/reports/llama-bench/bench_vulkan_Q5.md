| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |  99 |           pp512 |        268.16 ± 3.01 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |  99 |          pp2048 |        243.01 ± 1.53 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |  99 |           tg128 |         14.15 ± 0.51 |

build: 136887b66 (11221)
