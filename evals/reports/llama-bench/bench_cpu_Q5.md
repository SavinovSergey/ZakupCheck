| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |   0 |      16 |           pp512 |       164.19 ± 10.50 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |   0 |      16 |          pp2048 |        155.93 ± 3.63 |
| qwen3 8B Q5_K - Medium         |   5.44 GiB |     8.19 B | Vulkan     |   0 |      16 |           tg128 |          3.88 ± 0.41 |

build: 136887b66 (11221)
