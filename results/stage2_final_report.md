# Stage 2 final report: Qwen3-32B (uncapped, capped) and Gemma 2 27B (uncapped)

Written 2026-10-21 (UTC) on the Qwen machine (1x RTX PRO 6000 Blackwell, 96 GB).
- Full analysis: `stage2_summary.md` and the `stage2_*.csv` / `.png` files, from `scripts/analyze.py` over all three runs.
- Role-adoption check: `role_adoption_report.md`.
- Raw outputs: `Faruna01/persona-collapse-runs`, under `qwen/replay/`, `gemma/replay/` and `sensitivity/`.

## Runs

| Run | Machine | Generation | Wall time |
|---|---|---|---|
| Qwen3-32B uncapped | RTX PRO 6000 | vLLM 0.32.0, fp8 KV cache, `TRITON_ATTN`, gpu-mem 0.95, max_new_tokens 1024, T=0.7, top_p 0.9, seed 0 | 2026-10-20 12:38:32 → 23:48:02 (11.2 h) |
| Qwen3-32B capped (`layers_46:54-p0.25`) | RTX PRO 6000 | HF generate + `build_capping_steerer`, bf16, therapy + metacognitive, 10 conversations per domain (`--spread`), max_new_tokens 1024, `--hf-token-budget 65000` | 2026-10-20 23:48:02 → 2026-10-21 10:22:32 (10.6 h) |
| gemma-2-27b-it uncapped | A100-SXM4-80GB | vLLM (see the Gemma machine logs in the dataset) | – |

Notes on these settings:
- The fp8 KV cache only affects the sampled text. All projections use the bf16 model.
- FlashInfer's fp8 kernel cannot compile for sm_120 with CUDA 12.8, so the uncapped run used vLLM's Triton attention instead. In an 18-conversation test, speed matched bf16 to within 2–5% per token, and the KV cache was 2.48× larger.

## `project.py --check` (batched vs single-conversation projection)

| Run | Conversation checked | max \|single − batched\| | In normalised-score units |
|---|---|---|---|
| Qwen pilot (uncapped, 400 tokens) | coding_p0_t1/en | 0.0000 | 0.0000 |
| Qwen uncapped (full) | therapy_p5_t0/en | 0.0000 | 0.0000 |
| Qwen capped (full) | therapy_p5_t4/en | 0.0000 | 0.0000 |
| Gemma uncapped (full) | therapy_p5_t0/en | 0.0020 (raw projections ≈ 9,000) | 0.0000 |

- The Qwen uncapped and Gemma checks ran on the Gemma machine; their logs are `gemma/logs/stage2/*_project_check.log`.
- An earlier Qwen check printed `nan` in normalised units. Its log is kept as `qwen_project_check.nan_scores.log`. The final projection files contain no NaN scores.

## Row counts

| File | Rows |
|---|---|
| `qwen_uncapped.jsonl` (conversations) | 540 (180 × en/ha/yo) |
| `qwen_uncapped.proj.jsonl` / `.langid.jsonl` (turns) | 7,644 / 7,644 |
| `qwen_capped.jsonl` | 60 (2 domains × 10 × 3 languages) |
| `qwen_capped.proj.jsonl` / `.langid.jsonl` | 873 / 873 (291 per language) |
| `gemma_uncapped.jsonl` | 540 |
| `gemma_uncapped.proj.jsonl` / `.langid.jsonl` | 6,751 / 6,751 |
| `sensitivity/{qwen,gemma}_uncapped.jsonl` | 576 each |
| `sensitivity/role_judgments.jsonl` | 936 |

No conversation in either Qwen run hit the context limit. Gemma hit its 8,192-token limit in 198 conversations, 136 of them in Yoruba (see the token-budget table in `stage2_summary.md`).

## Capped vs uncapped (Qwen3-32B)

### Matched comparison
This uses the same 10 conversations per domain and language, the same turns, and pairs each conversation's mean score. It is my own calculation from the proj files, with a 2,000-sample bootstrap over conversations, seed 0. Turns are the capped run's comparable window.

| Domain | Language | Pairs | Turns | Uncapped score | Capped score | Capped − uncapped (95% CI) |
|---|---|---|---|---|---|---|
| therapy | en | 10 | 1-10 | -0.126 | 0.479 | +0.61 [+0.52, +0.69] |
| therapy | ha | 10 | 1-10 | 0.163 | 0.276 | +0.11 [+0.06, +0.19] |
| therapy | yo | 10 | 1-10 | 0.131 | 0.219 | +0.09 [+0.06, +0.12] |
| metacognitive | en | 10 | 1-15 | -0.507 | 0.051 | +0.56 [+0.43, +0.72] |
| metacognitive | ha | 10 | 1-15 | 0.132 | 0.233 | +0.10 [+0.06, +0.15] |
| metacognitive | yo | 10 | 1-15 | 0.124 | 0.244 | +0.12 [+0.09, +0.15] |

### As reported by `analyze.py`
This is unmatched: all 60 uncapped conversations against the 10 capped ones, each over its own comparable turns.

| Domain | Language | Uncapped mean score (n=60) | Uncapped drift vs own coding | Capped mean score (n=10) |
|---|---|---|---|---|
| therapy | en | -0.030 | -0.37 [-0.46, -0.28] | 0.479 |
| therapy | ha | 0.137 | -0.13 [-0.16, -0.11] | 0.276 |
| therapy | yo | 0.120 | -0.18 [-0.20, -0.16] | 0.219 |
| metacognitive | en | -0.552 | -0.89 [-0.96, -0.83] | 0.051 |
| metacognitive | ha | 0.131 | -0.14 [-0.16, -0.11] | 0.233 |
| metacognitive | yo | 0.107 | -0.20 [-0.22, -0.17] | 0.244 |

The capped run has no coding domain, so it has no per-language baseline and no drift. `analyze.py` now skips its drift chart (commit a32cbc4). The capped-vs-uncapped chart is `stage2_capping_Qwen3-32B.png`.

### Reading
- Capping moves the English score up by about 0.6 in both domains. In therapy, capped English (0.48) ends up above the uncapped English coding baseline (0.34).
- In Hausa and Yoruba the same cap moves the score by only about 0.1.
- The role-adoption check below shows the axis responds weakly to persona in ha/yo text. The small ha/yo capping effect is therefore at least partly the same weakness of the axis, not evidence that capping does less in those languages.
- Capped replies were 100% in the user's language, against 95–100% uncapped in the same two domains. They were also shorter in English: 9–22% hit max_new_tokens, against 29–38% uncapped.

## Uncapped drift across models
These come from `stage2_summary.md`. Drift is the score minus the same language's coding mean, shown with a 95% CI.

| Model | Domain | en | ha | yo |
|---|---|---|---|---|
| Qwen3-32B | therapy | -0.37 [-0.46, -0.28] | -0.13 [-0.16, -0.11] | -0.18 [-0.20, -0.16] |
| Qwen3-32B | metacognitive | -0.89 [-0.96, -0.83] | -0.14 [-0.16, -0.11] | -0.20 [-0.22, -0.17] |
| gemma-2-27b-it | therapy | -0.12 [-0.22, -0.03] | -0.91 [-1.11, -0.74] | +0.02 [-0.30, +0.33] |
| gemma-2-27b-it | metacognitive | -0.31 [-0.43, -0.20] | -0.70 [-0.93, -0.46] | -0.40 [-0.61, -0.20] |

- Gemma's coding baselines differ by 2.69 across languages (en 0.66, ha -2.03, yo -1.94).
- Its Yoruba therapy replies are only 77% in Yoruba, and 43–50 Yoruba conversations per domain stopped at the context limit.
- Gemma ha/yo drift mixes persona with reply language. See `role_adoption_report.md` and `stage2_reply_language_by_turn.csv`.

## Role adoption (sensitivity check, LLM judge)
The judge is `gemini-3.5-flash-lite`, over 936 single-turn replies. The full analysis is in `role_adoption_report.md`.

| Model | Language | Far: full | Far: somewhat | Far: none | Score, default | Score, far judged full (95% CI) | Score, far judged none | n far |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 96% | 4% | 0% | 0.49 | -0.88 [-0.97, -0.79] (n=115) |  | 120 |
| Qwen3-32B | ha | 34% | 18% | 48% | 0.31 | -0.35 [-0.51, -0.20] (n=41) | 0.05 | 120 |
| Qwen3-32B | yo | 54% | 12% | 34% | 0.30 | -0.27 [-0.42, -0.14] (n=65) | 0.08 | 120 |
| gemma-2-27b-it | en | 92% | 7% | 1% | 0.95 | -0.44 [-0.55, -0.34] (n=111) | 0.93 | 120 |
| gemma-2-27b-it | ha | 89% | 8% | 3% | -2.48 | -1.49 [-1.81, -1.20] (n=107) | -0.14 | 120 |
| gemma-2-27b-it | yo | 92% | 5% | 3% | -2.56 | -1.11 [-1.36, -0.89] (n=110) | -3.88 | 120 |

**What it means for (a) a less sensitive axis vs (b) less role-play.**
- **Qwen:** both explanations hold, and (a) is larger.
  - Qwen role-plays less in ha/yo (b).
  - But within Hausa and Yoruba replies, far roles judged "full" score almost the same as those judged "none": -0.06 vs +0.03 in Hausa, +0.07 vs +0.08 in Yoruba. Their gap from default is 15–27% of the English gap (a).
- **Gemma:** (b) is ruled out, because role adoption is about 90% in every language.
  - Its raw ha/yo scores are dominated by far roles switching to English.
  - Within each language, the default-minus-far gap is smaller than in English (a).
- These ha/yo judge labels are provisional until the native-speaker sheet `ratings/role_spotcheck_sheet.csv` is filled in.

## Artifact check (`verify_artifacts.py --machine qwen`)
- The upload pass with `--upload-missing` uploaded the Qwen pilot (`qwen/pilot/`) and this machine's logs (`qwen/logs/`).
- The final re-check reported 62 files present with matching sizes and nothing missing.
- Its own log is kept outside `/workspace/outputs/logs`, because writing it there would make it a new unsaved file.

```
Required for this machine's runs:
       ok  qwen/replay/qwen_uncapped.jsonl
       ok  qwen/replay/qwen_uncapped.log.jsonl
       ok  qwen/replay/qwen_uncapped.proj.jsonl
       ok  qwen/replay/qwen_uncapped.langid.jsonl
       ok  qwen/replay/qwen_capped.jsonl
       ok  qwen/replay/qwen_capped.log.jsonl
       ok  qwen/replay/qwen_capped.proj.jsonl
       ok  qwen/replay/qwen_capped.langid.jsonl

ALL REQUIRED FILES SAVED
```

The environment record is in `env/qwen-rtxpro6000/`, written by `scripts/save_env.sh`.
