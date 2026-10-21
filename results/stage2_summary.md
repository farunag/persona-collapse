# Stage 2 summary

Generated 2026-10-21T11:42:43.498599+00:00 by `scripts/analyze.py`. Inputs: `gemma_uncapped.proj.jsonl`, `qwen_uncapped.proj.jsonl`, `qwen_capped.proj.jsonl`, `gemma_uncapped.langid.jsonl`, `qwen_uncapped.langid.jsonl`, `qwen_capped.langid.jsonl`, `gemma_uncapped.jsonl`, `qwen_uncapped.jsonl`, `qwen_capped.jsonl`.
Bootstrap: 2000 resamples over conversations, seed 0. Comparable turns: reached by >= 10 conversations and by >= 80% of each language's conversations, in every language of the domain (unbroken from turn 1).
Score: normalised projection (1 = default Assistant vector, 0 = mean role vector). Drift: score minus the same language's coding-domain mean.

## Comparable turns

| Condition | Domain | Turns |
|---|---|---|
| Qwen3-32B uncapped | coding | 1-12 (12) |
| Qwen3-32B uncapped | metacognitive | 1-15 (15) |
| Qwen3-32B uncapped | therapy | 1-13 (13) |
| Qwen3-32B capped | metacognitive | 1-15 (15) |
| Qwen3-32B capped | therapy | 1-10 (10) |
| gemma-2-27b-it uncapped | coding | 1-8 (8) |
| gemma-2-27b-it uncapped | metacognitive | 1-8 (8) |
| gemma-2-27b-it uncapped | therapy | 1-7 (7) |

## Per-language coding baselines

| Condition | Language | Coding mean score | Conversations |
|---|---|---|---|
| Qwen3-32B uncapped | en | 0.339 | 60 |
| Qwen3-32B uncapped | ha | 0.270 | 60 |
| Qwen3-32B uncapped | yo | 0.302 | 60 |
| gemma-2-27b-it uncapped | en | 0.661 | 60 |
| gemma-2-27b-it uncapped | ha | -2.026 | 60 |
| gemma-2-27b-it uncapped | yo | -1.938 | 60 |

## Mean drift over comparable turns (95% CI)

| Condition | Domain | Language | Conversations | Mean score | Drift |
|---|---|---|---|---|---|
| Qwen3-32B uncapped | coding | en | 60 | 0.339 | +0.00 [-0.03, +0.03] |
| Qwen3-32B uncapped | coding | ha | 60 | 0.270 | +0.00 [-0.03, +0.03] |
| Qwen3-32B uncapped | coding | yo | 60 | 0.302 | +0.00 [-0.02, +0.02] |
| Qwen3-32B uncapped | therapy | en | 60 | -0.030 | -0.37 [-0.46, -0.28] |
| Qwen3-32B uncapped | therapy | ha | 60 | 0.137 | -0.13 [-0.16, -0.11] |
| Qwen3-32B uncapped | therapy | yo | 60 | 0.120 | -0.18 [-0.20, -0.16] |
| Qwen3-32B uncapped | metacognitive | en | 60 | -0.552 | -0.89 [-0.96, -0.83] |
| Qwen3-32B uncapped | metacognitive | ha | 60 | 0.131 | -0.14 [-0.16, -0.11] |
| Qwen3-32B uncapped | metacognitive | yo | 60 | 0.107 | -0.20 [-0.22, -0.17] |
| Qwen3-32B capped | therapy | en | 10 | 0.479 | – |
| Qwen3-32B capped | therapy | ha | 10 | 0.276 | – |
| Qwen3-32B capped | therapy | yo | 10 | 0.219 | – |
| Qwen3-32B capped | metacognitive | en | 10 | 0.051 | – |
| Qwen3-32B capped | metacognitive | ha | 10 | 0.233 | – |
| Qwen3-32B capped | metacognitive | yo | 10 | 0.244 | – |
| gemma-2-27b-it uncapped | coding | en | 60 | 0.661 | +0.00 [-0.09, +0.10] |
| gemma-2-27b-it uncapped | coding | ha | 60 | -2.026 | +0.00 [-0.26, +0.24] |
| gemma-2-27b-it uncapped | coding | yo | 60 | -1.938 | +0.00 [-0.26, +0.25] |
| gemma-2-27b-it uncapped | therapy | en | 60 | 0.540 | -0.12 [-0.22, -0.03] |
| gemma-2-27b-it uncapped | therapy | ha | 60 | -2.939 | -0.91 [-1.11, -0.74] |
| gemma-2-27b-it uncapped | therapy | yo | 60 | -1.922 | +0.02 [-0.30, +0.33] |
| gemma-2-27b-it uncapped | metacognitive | en | 60 | 0.350 | -0.31 [-0.43, -0.20] |
| gemma-2-27b-it uncapped | metacognitive | ha | 60 | -2.724 | -0.70 [-0.93, -0.46] |
| gemma-2-27b-it uncapped | metacognitive | yo | 60 | -2.339 | -0.40 [-0.61, -0.20] |

## Drift gap vs English (language drift minus English drift, 95% CI)

| Condition | Domain | Language | Gap |
|---|---|---|---|
| Qwen3-32B uncapped | coding | ha | +0.00 [-0.05, +0.04] |
| Qwen3-32B uncapped | coding | yo | +0.00 [-0.04, +0.04] |
| Qwen3-32B uncapped | metacognitive | ha | +0.75 [+0.69, +0.82] |
| Qwen3-32B uncapped | metacognitive | yo | +0.70 [+0.63, +0.76] |
| Qwen3-32B uncapped | therapy | ha | +0.24 [+0.15, +0.33] |
| Qwen3-32B uncapped | therapy | yo | +0.19 [+0.10, +0.28] |
| gemma-2-27b-it uncapped | coding | ha | +0.00 [-0.28, +0.26] |
| gemma-2-27b-it uncapped | coding | yo | +0.00 [-0.27, +0.27] |
| gemma-2-27b-it uncapped | metacognitive | ha | -0.39 [-0.64, -0.13] |
| gemma-2-27b-it uncapped | metacognitive | yo | -0.09 [-0.33, +0.13] |
| gemma-2-27b-it uncapped | therapy | ha | -0.79 [-1.02, -0.60] |
| gemma-2-27b-it uncapped | therapy | yo | +0.14 [-0.19, +0.48] |

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
| Qwen3-32B capped | en | metacognitive | 150 | 100.0% | 150 / 0 / 0 / 0 |
| Qwen3-32B capped | en | therapy | 141 | 100.0% | 141 / 0 / 0 / 0 |
| Qwen3-32B capped | ha | metacognitive | 150 | 100.0% | 0 / 150 / 0 / 0 |
| Qwen3-32B capped | ha | therapy | 141 | 100.0% | 0 / 141 / 0 / 0 |
| Qwen3-32B capped | yo | metacognitive | 150 | 100.0% | 0 / 0 / 150 / 0 |
| Qwen3-32B capped | yo | therapy | 141 | 100.0% | 0 / 0 / 141 / 0 |
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
| Qwen3-32B capped | en | metacognitive | 743 | 9.3% | 0 | 13864 |
| Qwen3-32B capped | en | therapy | 724 | 22.0% | 0 | 14279 |
| Qwen3-32B capped | ha | metacognitive | 934 | 73.3% | 0 | 16102 |
| Qwen3-32B capped | ha | therapy | 918 | 67.4% | 0 | 15513 |
| Qwen3-32B capped | yo | metacognitive | 980 | 81.3% | 0 | 15931 |
| Qwen3-32B capped | yo | therapy | 794 | 61.7% | 0 | 15391 |
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

- gemma-2-27b-it uncapped: coding baselines differ across languages by 2.69 (yo -1.94, ha -2.03, en 0.66); language alone shifts the projection, so compare drift, not raw scores.
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
- Qwen3-32B capped en/therapy: 22% of replies cut at max_new_tokens.
- Qwen3-32B capped ha/metacognitive: 73% of replies cut at max_new_tokens.
- Qwen3-32B capped ha/therapy: 67% of replies cut at max_new_tokens.
- Qwen3-32B capped yo/metacognitive: 81% of replies cut at max_new_tokens.
- Qwen3-32B capped yo/therapy: 62% of replies cut at max_new_tokens.
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
- `stage2_capping_Qwen3-32B.png`
- `stage2_reply_language_Qwen3-32B_uncapped.png`
- `stage2_reply_language_Qwen3-32B_capped.png`
- `stage2_reply_language_gemma-2-27b-it_uncapped.png`

## Provenance

- Qwen/Qwen3-32B revision 9216db5781bf21249d130ec9da846c4624c16137, generation seed 0
- google/gemma-2-27b-it revision aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0, generation seed 0
- projection libraries: {'torch': '2.9.0+cu128', 'transformers': '4.57.5', 'assistant_axis': '/workspace/assistant-axis/assistant_axis', 'python': '3.11.17', 'gpu': 'NVIDIA A100-SXM4-80GB'}
- projection libraries: {'torch': '2.9.0+cu128', 'transformers': '4.57.5', 'assistant_axis': '/workspace/assistant-axis/assistant_axis', 'python': '3.11.17', 'gpu': 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'}
