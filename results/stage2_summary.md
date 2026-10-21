# Stage 2 summary

Generated 2026-10-21T08:19:04.558507+00:00 by `scripts/analyze.py`. Inputs: `gemma_uncapped.proj.jsonl`, `qwen_uncapped.proj.jsonl`, `gemma_uncapped.langid.jsonl`, `qwen_uncapped.langid.jsonl`, `gemma_uncapped.jsonl`, `qwen_uncapped.jsonl`.
Bootstrap: 2000 resamples over conversations, seed 0. Comparable turns: reached by >= 10 conversations in every language of the domain.
Score: normalised projection (1 = default Assistant vector, 0 = mean role vector). Drift: score minus the same language's coding-domain mean.

## Comparable turns

| Condition | Domain | Turns |
|---|---|---|
| Qwen3-32B uncapped | coding | 1-15 (15) |
| Qwen3-32B uncapped | metacognitive | 1-15 (15) |
| Qwen3-32B uncapped | therapy | 1-15 (15) |
| gemma-2-27b-it uncapped | coding | 1-12 (12) |
| gemma-2-27b-it uncapped | metacognitive | 1-15 (15) |
| gemma-2-27b-it uncapped | therapy | 1-14 (14) |

## Per-language coding baselines

| Condition | Language | Coding mean score | Conversations |
|---|---|---|---|
| Qwen3-32B uncapped | en | 0.333 | 60 |
| Qwen3-32B uncapped | ha | 0.265 | 60 |
| Qwen3-32B uncapped | yo | 0.298 | 60 |
| gemma-2-27b-it uncapped | en | 0.715 | 60 |
| gemma-2-27b-it uncapped | ha | -1.979 | 60 |
| gemma-2-27b-it uncapped | yo | -1.888 | 60 |

## Mean drift over comparable turns (95% CI)

| Condition | Domain | Language | Conversations | Mean score | Drift |
|---|---|---|---|---|---|
| Qwen3-32B uncapped | coding | en | 60 | 0.333 | +0.00 [-0.03, +0.03] |
| Qwen3-32B uncapped | coding | ha | 60 | 0.265 | +0.00 [-0.03, +0.03] |
| Qwen3-32B uncapped | coding | yo | 60 | 0.298 | +0.00 [-0.02, +0.02] |
| Qwen3-32B uncapped | therapy | en | 60 | -0.052 | -0.38 [-0.48, -0.30] |
| Qwen3-32B uncapped | therapy | ha | 60 | 0.134 | -0.13 [-0.16, -0.10] |
| Qwen3-32B uncapped | therapy | yo | 60 | 0.118 | -0.18 [-0.20, -0.16] |
| Qwen3-32B uncapped | metacognitive | en | 60 | -0.552 | -0.88 [-0.95, -0.83] |
| Qwen3-32B uncapped | metacognitive | ha | 60 | 0.131 | -0.13 [-0.16, -0.11] |
| Qwen3-32B uncapped | metacognitive | yo | 60 | 0.107 | -0.19 [-0.21, -0.17] |
| gemma-2-27b-it uncapped | coding | en | 60 | 0.715 | +0.00 [-0.08, +0.08] |
| gemma-2-27b-it uncapped | coding | ha | 60 | -1.979 | +0.00 [-0.25, +0.26] |
| gemma-2-27b-it uncapped | coding | yo | 60 | -1.888 | +0.00 [-0.24, +0.25] |
| gemma-2-27b-it uncapped | therapy | en | 60 | 0.580 | -0.13 [-0.22, -0.05] |
| gemma-2-27b-it uncapped | therapy | ha | 60 | -2.672 | -0.69 [-0.92, -0.50] |
| gemma-2-27b-it uncapped | therapy | yo | 60 | -1.766 | +0.12 [-0.19, +0.42] |
| gemma-2-27b-it uncapped | metacognitive | en | 60 | 0.237 | -0.48 [-0.63, -0.35] |
| gemma-2-27b-it uncapped | metacognitive | ha | 60 | -2.455 | -0.48 [-0.70, -0.26] |
| gemma-2-27b-it uncapped | metacognitive | yo | 60 | -2.159 | -0.27 [-0.46, -0.09] |

## Drift gap vs English (language drift minus English drift, 95% CI)

| Condition | Domain | Language | Gap |
|---|---|---|---|
| Qwen3-32B uncapped | coding | ha | +0.00 [-0.04, +0.04] |
| Qwen3-32B uncapped | coding | yo | +0.00 [-0.03, +0.04] |
| Qwen3-32B uncapped | metacognitive | ha | +0.75 [+0.68, +0.82] |
| Qwen3-32B uncapped | metacognitive | yo | +0.69 [+0.63, +0.76] |
| Qwen3-32B uncapped | therapy | ha | +0.25 [+0.16, +0.34] |
| Qwen3-32B uncapped | therapy | yo | +0.20 [+0.12, +0.30] |
| gemma-2-27b-it uncapped | coding | ha | +0.00 [-0.26, +0.26] |
| gemma-2-27b-it uncapped | coding | yo | +0.00 [-0.26, +0.26] |
| gemma-2-27b-it uncapped | metacognitive | ha | +0.00 [-0.25, +0.26] |
| gemma-2-27b-it uncapped | metacognitive | yo | +0.21 [-0.02, +0.44] |
| gemma-2-27b-it uncapped | therapy | ha | -0.56 [-0.77, -0.36] |
| gemma-2-27b-it uncapped | therapy | yo | +0.26 [-0.05, +0.57] |

## Reply language (all turns)

| Condition | Language | Domain | Replies | In user's language | en / ha / yo / other |
|---|---|---|---|---|---|
| Qwen3-32B uncapped | en | coding | 817 | 93.5% | 764 / 0 / 31 / 22 |
| Qwen3-32B uncapped | en | metacognitive | 900 | 98.3% | 885 / 0 / 4 / 11 |
| Qwen3-32B uncapped | en | therapy | 831 | 100.0% | 831 / 0 / 0 / 0 |
| Qwen3-32B uncapped | ha | coding | 817 | 85.1% | 3 / 695 / 4 / 115 |
| Qwen3-32B uncapped | ha | metacognitive | 900 | 95.2% | 1 / 857 / 0 / 42 |
| Qwen3-32B uncapped | ha | therapy | 831 | 96.9% | 0 / 805 / 0 / 26 |
| Qwen3-32B uncapped | yo | coding | 817 | 96.8% | 18 / 0 / 791 / 8 |
| Qwen3-32B uncapped | yo | metacognitive | 900 | 98.8% | 0 / 0 / 889 / 11 |
| Qwen3-32B uncapped | yo | therapy | 831 | 97.7% | 2 / 0 / 812 / 17 |
| gemma-2-27b-it uncapped | en | coding | 788 | 99.9% | 787 / 0 / 0 / 1 |
| gemma-2-27b-it uncapped | en | metacognitive | 900 | 99.3% | 894 / 0 / 0 / 6 |
| gemma-2-27b-it uncapped | en | therapy | 823 | 97.6% | 803 / 0 / 0 / 20 |
| gemma-2-27b-it uncapped | ha | coding | 737 | 93.8% | 46 / 691 / 0 / 0 |
| gemma-2-27b-it uncapped | ha | metacognitive | 885 | 95.7% | 31 / 847 / 0 / 7 |
| gemma-2-27b-it uncapped | ha | therapy | 800 | 98.9% | 0 / 791 / 0 / 9 |
| gemma-2-27b-it uncapped | yo | coding | 566 | 91.2% | 50 / 0 / 516 / 0 |
| gemma-2-27b-it uncapped | yo | metacognitive | 649 | 98.2% | 3 / 0 / 637 / 9 |
| gemma-2-27b-it uncapped | yo | therapy | 603 | 77.4% | 134 / 0 / 467 / 2 |

## Token budget

| Condition | Language | Domain | Mean reply tokens | Hit max_new_tokens | Context stops | Max prompt tokens |
|---|---|---|---|---|---|---|
| Qwen3-32B uncapped | en | coding | 918 | 54.5% | 0 | 15142 |
| Qwen3-32B uncapped | en | metacognitive | 852 | 38.0% | 0 | 15760 |
| Qwen3-32B uncapped | en | therapy | 777 | 29.2% | 0 | 14716 |
| Qwen3-32B uncapped | ha | coding | 1019 | 97.3% | 0 | 16415 |
| Qwen3-32B uncapped | ha | metacognitive | 962 | 82.9% | 0 | 17112 |
| Qwen3-32B uncapped | ha | therapy | 932 | 72.3% | 0 | 15784 |
| Qwen3-32B uncapped | yo | coding | 1022 | 97.9% | 0 | 16664 |
| Qwen3-32B uncapped | yo | metacognitive | 1000 | 93.3% | 0 | 17536 |
| Qwen3-32B uncapped | yo | therapy | 963 | 88.9% | 0 | 16207 |
| gemma-2-27b-it uncapped | en | coding | 414 | 0.1% | 13 | 7063 |
| gemma-2-27b-it uncapped | en | metacognitive | 218 | 0.0% | 0 | 6098 |
| gemma-2-27b-it uncapped | en | therapy | 326 | 0.0% | 6 | 7096 |
| gemma-2-27b-it uncapped | ha | coding | 458 | 0.9% | 24 | 7166 |
| gemma-2-27b-it uncapped | ha | metacognitive | 265 | 1.4% | 5 | 7147 |
| gemma-2-27b-it uncapped | ha | therapy | 339 | 0.2% | 14 | 7158 |
| gemma-2-27b-it uncapped | yo | coding | 680 | 15.2% | 50 | 7166 |
| gemma-2-27b-it uncapped | yo | metacognitive | 548 | 22.8% | 43 | 7165 |
| gemma-2-27b-it uncapped | yo | therapy | 594 | 28.4% | 43 | 7137 |

## Flags (check before interpreting)

- gemma-2-27b-it uncapped: coding baselines differ across languages by 2.69 (yo -1.89, ha -1.98, en 0.71); language alone shifts the projection, so compare drift, not raw scores.
- Qwen3-32B uncapped ha/coding: only 85% of replies in the user's language.
- gemma-2-27b-it uncapped yo/therapy: only 77% of replies in the user's language.
- Qwen3-32B uncapped en/coding: 54% of replies cut at max_new_tokens.
- Qwen3-32B uncapped en/metacognitive: 38% of replies cut at max_new_tokens.
- Qwen3-32B uncapped en/therapy: 29% of replies cut at max_new_tokens.
- Qwen3-32B uncapped ha/coding: 97% of replies cut at max_new_tokens.
- Qwen3-32B uncapped ha/metacognitive: 83% of replies cut at max_new_tokens.
- Qwen3-32B uncapped ha/therapy: 72% of replies cut at max_new_tokens.
- Qwen3-32B uncapped yo/coding: 98% of replies cut at max_new_tokens.
- Qwen3-32B uncapped yo/metacognitive: 93% of replies cut at max_new_tokens.
- Qwen3-32B uncapped yo/therapy: 89% of replies cut at max_new_tokens.
- gemma-2-27b-it uncapped en/coding: 13 conversations stopped at the context limit.
- gemma-2-27b-it uncapped en/therapy: 6 conversations stopped at the context limit.
- gemma-2-27b-it uncapped ha/coding: 24 conversations stopped at the context limit.
- gemma-2-27b-it uncapped ha/metacognitive: 5 conversations stopped at the context limit.
- gemma-2-27b-it uncapped ha/therapy: 14 conversations stopped at the context limit.
- gemma-2-27b-it uncapped yo/coding: 50 conversations stopped at the context limit.
- gemma-2-27b-it uncapped yo/metacognitive: 43 conversations stopped at the context limit.
- gemma-2-27b-it uncapped yo/metacognitive: 23% of replies cut at max_new_tokens.
- gemma-2-27b-it uncapped yo/therapy: 43 conversations stopped at the context limit.
- gemma-2-27b-it uncapped yo/therapy: 28% of replies cut at max_new_tokens.

## Charts

- `stage2_drift_Qwen3-32B_uncapped.png`
- `stage2_drift_gemma-2-27b-it_uncapped.png`
- `stage2_reply_language_Qwen3-32B_uncapped.png`
- `stage2_reply_language_gemma-2-27b-it_uncapped.png`

## Provenance

- Qwen/Qwen3-32B revision 9216db5781bf21249d130ec9da846c4624c16137, generation seed 0
- google/gemma-2-27b-it revision aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0, generation seed 0
- projection libraries: {'torch': '2.9.0+cu128', 'transformers': '4.57.5', 'assistant_axis': '/workspace/assistant-axis/assistant_axis', 'python': '3.11.17', 'gpu': 'NVIDIA A100-SXM4-80GB'}
