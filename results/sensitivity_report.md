# Axis sensitivity check: results

Run 2026-10-21 (UTC) on the Gemma machine (1x A100-SXM4-80GB). Prompts: `data/source/sensitivity.jsonl`, 576 single-turn prompts (12 user messages × 16 roles × 3 languages).
Generation used vLLM with `--max-new-tokens 300`, seed 0 and the same revisions as stage 2. All 576 prompts were replayed, projected and language-tagged for both models, with no errors.
Full output from `analyze_sensitivity.py`: `sensitivity_summary.md`, `sensitivity_separation.csv` and `sensitivity_roles.csv`.

| Model | Language | Default | Near roles | Far roles | Separation (95% CI) | vs English | Replies in user's language |
|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 0.49 | 0.48 | -0.89 | 1.37 [1.16, 1.57] | 1.00 | 99% |
| Qwen3-32B | ha | 0.31 | 0.37 | -0.09 | 0.40 [0.33, 0.47] | 0.29 | 86% |
| Qwen3-32B | yo | 0.30 | 0.44 | -0.10 | 0.41 [0.33, 0.48] | 0.30 | 81% |
| gemma-2-27b-it | en | 0.95 | 0.93 | -0.38 | 1.33 [1.10, 1.57] | 1.00 | 99% |
| gemma-2-27b-it | ha | -2.48 | -1.56 | -1.54 | -0.94 [-1.67, -0.24] | -0.71 | 50% |
| gemma-2-27b-it | yo | -2.56 | -0.32 | -1.32 | -1.24 [-2.00, -0.39] | -0.93 | 31% |

## What stands out

1. **In Hausa and Yoruba, the axis separates the roles much less well than in English.** For Qwen the separation is about 30% of the English value, with a clean CI. Low Qwen drift in ha/yo in stage 2 is therefore weak evidence that Qwen stays in assistant mode: the axis may just respond less in those languages.
2. **For Gemma the separation in Hausa and Yoruba is negative, and reply language confounds it.** Far-role prompts usually get English replies: 78 of 120 in Hausa and 96 of 120 in Yoruba. Gemma's score depends mainly on the language of the reply. Mean scores, from my breakdown of the proj and langid files (not part of `analyze_sensitivity.py`):

   | Gemma, user language | Group | Reply in user's language | Reply in another language |
   |---|---|---|---|
   | ha | default | −2.94 (n=32) | +1.15 (n=4) |
   | ha | far | −3.75 (n=42) | −0.35 (n=78) |
   | yo | default | −3.71 (n=27) | +0.92 (n=9) |
   | yo | far | −4.05 (n=24) | −0.64 (n=96) |

   Among replies in the user's language, default still scores above far, by about 0.8 for Hausa and 0.3 for Yoruba. These differences are unpaired and have no CI. The negative headline numbers come from far roles switching to English, which moves the score up by about 3–4 units. For Gemma, any ha/yo projection mixes persona with reply language. The stage 2 Gemma drift in ha/yo should be read alongside reply language per turn (`stage2_reply_language_by_turn.csv`).
3. Qwen shows a weaker form of the same effect. Its far-role replies in another language score lower, not higher: −0.61 in Hausa and −0.87 in Yoruba, compared with +0.02 and +0.08 for in-language replies.
4. Near roles (analyst, consultant, evaluator) score about the same as default in English, as expected. For Gemma in Yoruba they score well above default (−0.32 vs −2.56), again because 27 of 36 near-role replies are not in Yoruba.

## Run notes

- The first Qwen vLLM start failed. At max length 32,768 it needs 8.0 GiB of KV cache, and only 7.6 GiB was free on the 80 GB A100. I re-ran Qwen with `--gpu-memory-utilization 0.93`, which does not change generation. Gemma ran with the given command.
- `project.py` ran with `HF_HUB_OFFLINE=1`. Both axes, including the role vectors, were confirmed cached, so the scores contain no NaNs.
- I uploaded the outputs to `sensitivity/` and the run logs to `gemma/logs/stage2/` in `Faruna01/persona-collapse-runs`. All 43 files are present with matching sizes. `verify_artifacts.py --machine gemma` reports ALL REQUIRED FILES SAVED.
