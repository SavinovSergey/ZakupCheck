| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |   0 |      16 |           pp512 |        170.03 ± 6.45 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |   0 |      16 |          pp2048 |        160.32 ± 1.51 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     |   0 |      16 |           tg128 |          4.05 ± 0.79 |

build: 136887b66 (11221)
