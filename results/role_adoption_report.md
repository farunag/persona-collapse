# Role adoption in the axis sensitivity check: less sensitive axis, or less role-play?

Run 2026-10-21 (UTC) on the Qwen machine (1x RTX PRO 6000).

## Inputs
- The sensitivity replays from `Faruna01/persona-collapse-runs` `sensitivity/`: single-turn replies, 300 new tokens, seed 0.
- The judge was `gemini-3.5-flash-lite`, run by `scripts/judge_roles.py` at temperature 0.
- It rated the 13 default and far roles × 12 messages × 3 languages × 2 models, which is 936 replies. All were judged, with no retries and no failures.
- Judgments are in `sensitivity/role_judgments.jsonl` in the dataset (344,802 bytes).
- A 20-reply pilot was read first. Its labels matched the replies, including a degenerate, repetitive Qwen Yoruba reply rated "none" and Gemma answering Hausa in English, which the judge tagged as "en".
- `scripts/analyze_judgments.py` wrote `role_adoption.md`, `role_adoption.csv` and the native-speaker sheet `ratings/role_spotcheck_sheet.csv`.

## Role adoption and axis scores
Scores are normalised: 1 = default Assistant, 0 = mean role vector. They are taken at turn 1.

| Model | Language | Far: full | Far: somewhat | Far: none | Score, default | Score, far judged full (95% CI) | Score, far judged none | n far |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 96% | 4% | 0% | 0.49 | -0.88 [-0.97, -0.79] (n=115) |  | 120 |
| Qwen3-32B | ha | 34% | 18% | 48% | 0.31 | -0.35 [-0.51, -0.20] (n=41) | 0.05 | 120 |
| Qwen3-32B | yo | 54% | 12% | 34% | 0.30 | -0.27 [-0.42, -0.14] (n=65) | 0.08 | 120 |
| gemma-2-27b-it | en | 92% | 7% | 1% | 0.95 | -0.44 [-0.55, -0.34] (n=111) | 0.93 | 120 |
| gemma-2-27b-it | ha | 89% | 8% | 3% | -2.48 | -1.49 [-1.81, -1.20] (n=107) | -0.14 | 120 |
| gemma-2-27b-it | yo | 92% | 5% | 3% | -2.56 | -1.11 [-1.36, -0.89] (n=110) | -3.88 | 120 |

## The same scores split by reply language
This is my own breakdown, not part of `analyze_judgments.py`. A reply counts as in the user's language when the judge's `reply_language` equals the prompt language. Cells show the mean score and n; the differences are unpaired and have no CI.

| Model | Lang | Group | Reply in user's language | Reply in another language |
|---|---|---|---|---|
| Qwen3-32B | ha | default | +0.31 (35) | +0.49 (1) |
| Qwen3-32B | ha | far, judged full | **-0.06 (25)** | -0.81 (16) |
| Qwen3-32B | ha | far, judged none | +0.03 (54) | +0.36 (3) |
| Qwen3-32B | yo | default | +0.27 (33) | +0.60 (3) |
| Qwen3-32B | yo | far, judged full | **+0.07 (41)** | -0.85 (24) |
| Qwen3-32B | yo | far, judged none | +0.08 (41) | – |
| gemma-2-27b-it | ha | default | -2.94 (32) | +1.15 (4) |
| gemma-2-27b-it | ha | far, judged full | -3.79 (34) | -0.42 (73) |
| gemma-2-27b-it | yo | default | -3.71 (27) | +0.92 (9) |
| gemma-2-27b-it | yo | far, judged full | -4.04 (15) | -0.64 (95) |

In English, default minus far-judged-full is 1.37 for Qwen and 1.39 for Gemma.

## What it implies

**Qwen: both effects are real, but (a) a less sensitive axis is the larger one.**
- (b) Qwen plays the far roles less often in Hausa and Yoruba: 34% and 54% "full", against 96% in English. Part of the smaller ha/yo separation is therefore real behaviour.
- (a) Even when the judge says Qwen fully plays the role, a reply in Hausa or Yoruba barely moves on the axis.
  - The gap from default is +0.31 → -0.06 for Hausa and +0.27 → +0.07 for Yoruba, a separation of about 0.37 and 0.20. That is 15–27% of English's 1.37.
  - Within ha/yo text, replies judged "full" and "none" score almost the same: -0.06 vs +0.03 in Hausa, +0.07 vs +0.08 in Yoruba. The axis hardly tells role-play from non-role-play inside these languages.
  - The far-full replies that do score low (-0.81, -0.85) are mostly the ones written in another language, mainly English.
- For stage 2, small Qwen drift in Hausa and Yoruba cannot be read as "Qwen stays the Assistant". The axis is a weak instrument in those languages.

**Gemma: (b) is ruled out, and (a) plus reply language explains the result.**
- Gemma plays the far roles as often in Hausa and Yoruba (89%, 92%) as in English (92%), so less role-play is not the explanation.
- The raw far-full scores sit above default in ha/yo, the wrong direction, because 73 of 107 Hausa and 95 of 110 Yoruba far-full replies are not in the user's language. Switching to English lifts the score by about 3–4 units.
- Among in-language replies the sign is right, but the gap is small: -0.85 in Hausa and -0.33 in Yoruba, against -1.39 in English.

## Caveats
- The judge is an LLM rating Hausa and Yoruba text. It may be lenient about what counts as "full" role-play in weak or partly incoherent replies, and several Qwen ha/yo replies are degenerate. The native-speaker sheet `ratings/role_spotcheck_sheet.csv` (40 rows, judge labels hidden) is the check on this. Until it is filled in, treat the ha/yo role-adoption rates as provisional.
- These are single-turn replies of 300 tokens. Stage 2 is multi-turn, up to 1,024 tokens.
