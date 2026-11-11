# Drift with replies in the user's language only

| Model | Turns used | Language | Domain | Conversations | Turns kept | Drift [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-32B | all | en | therapy | 60 | 100% | -0.37 [-0.46, -0.29] |
| Qwen3-32B | all | en | metacognitive | 60 | 100% | -0.89 [-0.95, -0.83] |
| Qwen3-32B | all | ha | therapy | 60 | 100% | -0.13 [-0.15, -0.12] |
| Qwen3-32B | all | ha | metacognitive | 60 | 100% | -0.14 [-0.15, -0.13] |
| Qwen3-32B | all | yo | therapy | 60 | 100% | -0.18 [-0.20, -0.17] |
| Qwen3-32B | all | yo | metacognitive | 60 | 100% | -0.20 [-0.21, -0.18] |
| Qwen3-32B | in-language | en | therapy | 60 | 100% | -0.37 [-0.45, -0.28] |
| Qwen3-32B | in-language | en | metacognitive | 60 | 98% | -0.89 [-0.95, -0.83] |
| Qwen3-32B | in-language | ha | therapy | 60 | 97% | -0.17 [-0.18, -0.15] |
| Qwen3-32B | in-language | ha | metacognitive | 60 | 95% | -0.17 [-0.18, -0.15] |
| Qwen3-32B | in-language | yo | therapy | 60 | 98% | -0.18 [-0.20, -0.17] |
| Qwen3-32B | in-language | yo | metacognitive | 60 | 99% | -0.20 [-0.21, -0.18] |
| Qwen3-32B | in-language prose | en | therapy | 60 | 100% | -0.34 [-0.42, -0.25] |
| Qwen3-32B | in-language prose | en | metacognitive | 60 | 100% | -0.86 [-0.92, -0.80] |
| Qwen3-32B | in-language prose | ha | therapy | 60 | 100% | -0.13 [-0.15, -0.12] |
| Qwen3-32B | in-language prose | ha | metacognitive | 60 | 100% | -0.13 [-0.15, -0.12] |
| Qwen3-32B | in-language prose | yo | therapy | 60 | 100% | -0.16 [-0.18, -0.15] |
| Qwen3-32B | in-language prose | yo | metacognitive | 60 | 100% | -0.18 [-0.19, -0.16] |
| gemma-2-27b-it | all | en | therapy | 60 | 100% | -0.12 [-0.19, -0.06] |
| gemma-2-27b-it | all | en | metacognitive | 60 | 100% | -0.31 [-0.40, -0.22] |
| gemma-2-27b-it | all | ha | therapy | 60 | 100% | -0.91 [-0.97, -0.86] |
| gemma-2-27b-it | all | ha | metacognitive | 60 | 100% | -0.70 [-0.82, -0.53] |
| gemma-2-27b-it | all | yo | therapy | 60 | 100% | +0.02 [-0.25, +0.29] |
| gemma-2-27b-it | all | yo | metacognitive | 60 | 100% | -0.40 [-0.50, -0.30] |
| gemma-2-27b-it | in-language | en | therapy | 60 | 99% | -0.11 [-0.18, -0.05] |
| gemma-2-27b-it | in-language | en | metacognitive | 60 | 99% | -0.30 [-0.40, -0.21] |
| gemma-2-27b-it | in-language | ha | therapy | 60 | 100% | -0.77 [-0.82, -0.71] |
| gemma-2-27b-it | in-language | ha | metacognitive | 58 | 96% | -0.66 [-0.71, -0.61] |
| gemma-2-27b-it | in-language | yo | therapy | 52 | 83% | -0.34 [-0.48, -0.21] |
| gemma-2-27b-it | in-language | yo | metacognitive | 60 | 98% | -0.31 [-0.41, -0.21] |
| gemma-2-27b-it | in-language prose | en | therapy | 60 | 100% | -0.44 [-0.50, -0.38] |
| gemma-2-27b-it | in-language prose | en | metacognitive | 60 | 100% | -0.64 [-0.74, -0.55] |
| gemma-2-27b-it | in-language prose | ha | therapy | 60 | 100% | -0.49 [-0.55, -0.44] |
| gemma-2-27b-it | in-language prose | ha | metacognitive | 58 | 100% | -0.39 [-0.44, -0.34] |
| gemma-2-27b-it | in-language prose | yo | therapy | 52 | 100% | -0.01 [-0.15, +0.11] |
| gemma-2-27b-it | in-language prose | yo | metacognitive | 60 | 100% | +0.02 [-0.08, +0.12] |

## Gemma per-turn drift, in-language replies (mean score minus that language's coding baseline)

| Language | Domain | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 |
|---|---|---|---|---|---|---|---|---|---|
| en | therapy | +0.30 (60) | -0.11 (60) | -0.25 (60) | -0.19 (59) | -0.18 (60) | -0.15 (59) | -0.22 (59) | – |
| en | metacognitive | +0.45 (59) | -0.13 (59) | -0.42 (60) | -0.40 (60) | -0.39 (60) | -0.55 (60) | -0.40 (59) | -0.55 (59) |
| ha | therapy | -0.95 (60) | -1.01 (60) | -0.92 (60) | -0.76 (60) | -0.64 (60) | -0.56 (59) | -0.51 (59) | – |
| ha | metacognitive | -0.59 (57) | -0.88 (58) | -0.94 (58) | -0.80 (58) | -0.71 (58) | -0.59 (58) | -0.42 (58) | -0.35 (56) |
| yo | therapy | -1.30 (48) | -0.85 (52) | -0.35 (51) | -0.14 (50) | +0.07 (50) | +0.20 (50) | +0.24 (49) | – |
| yo | metacognitive | -1.38 (58) | -0.96 (57) | -0.51 (59) | -0.21 (59) | +0.03 (59) | +0.20 (59) | +0.26 (59) | +0.19 (47) |

## In-language coding turns with and without code blocks

| Model | Language | With code: turns, mean score | Without code: turns, mean score |
|---|---|---|---|
| Qwen3-32B | en | 565, +0.34 | 88, +0.29 |
| Qwen3-32B | ha | 357, +0.33 | 241, +0.26 |
| Qwen3-32B | yo | 452, +0.32 | 223, +0.26 |
| gemma-2-27b-it | en | 331, +0.52 | 148, +0.97 |
| gemma-2-27b-it | ha | 268, -2.00 | 183, -2.46 |
| gemma-2-27b-it | yo | 183, -1.82 | 257, -2.31 |
