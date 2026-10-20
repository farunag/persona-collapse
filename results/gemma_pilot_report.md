# Gemma pilot report (uncapped, 2026-10-20)

Generation only. Projection, language ID and analysis have **not** been run on these
outputs yet, so this report has no persona-axis or language-choice results.

## Run

| | |
|---|---|
| Model | `google/gemma-2-27b-it` (revision `aaf20e6b9f4c`) |
| Backend | vLLM 0.32.0 (no fall-back), default attention backend, KV cache dtype `auto` |
| Setting | uncapped, seed 0, temperature 0.7, top-p 0.9 |
| Limits | max_new_tokens 1024, max_context 8192 |
| Scope | `--limit 2`: 2 conversations per domain (coding, therapy, metacognitive) x 3 languages (en, ha, yo) |
| Output | `/workspace/outputs/pilot_gemma/gemma_uncapped.jsonl` (18 conversations) |
| Wall time | 926.5 s (~15.5 min), finished 18:19:30 UTC |

## Conversation completion

- 18 of 18 conversations written; 223 assistant replies, 101,527 reply tokens in total.
- Scripted length: 15 user turns, except `coding_p0_t1` (10 turns).
- 11 conversations completed their full script: 9 ran all 15 turns, and the 2 `coding_p0_t1`
  conversations in en and ha finished their 10 turns.
- 7 conversations were cut off early by the context limit (prompt + 1024 > 8192). Six were
  Yoruba and one was Hausa. No English conversation was cut off.

**Correction to my earlier chat summary:** I said 9 conversations stopped early because of
the context limit. The correct number is 7. The other 2 short conversations (en and ha
`coding_p0_t1`) finished normally, since that script has only 10 turns.

### Context stops per language

| Lang | Conversation | Scripted turns | Replies completed | Stopped at turn |
|---|---|---|---|---|
| en | none | | | |
| ha | therapy_p0_t0 | 15 | 13 | 14 |
| yo | therapy_p0_t1 | 15 | 7 | 8 |
| yo | metacognitive_p0_t0 | 15 | 8 | 9 |
| yo | metacognitive_p0_t1 | 15 | 8 | 9 |
| yo | coding_p0_t1 | 10 | 9 | 10 |
| yo | coding_p0_t0 | 15 | 10 | 11 |
| yo | therapy_p0_t0 | 15 | 13 | 14 |

All 6 Yoruba conversations hit the limit. The median Yoruba stop came at turn 9.5.

## Reply lengths (tokens)

| Lang | Replies | Median | Mean | Hit 1024 cap |
|---|---|---|---|---|
| en | 85 | 332 | 342 | 0 |
| ha | 83 | 401 | 396 | 0 |
| yo | 55 | 736 | 720 | 12 |
| all | 223 | 422 | 455 | 12 |

Median reply length per conversation:

| Conversation | en | ha | yo |
|---|---|---|---|
| coding_p0_t0 | 258 | 367 | 590 |
| coding_p0_t1 | 521 | 600 | 794 |
| metacognitive_p0_t0 | 308 | 221 | 844 |
| metacognitive_p0_t1 | 195 | 415 | 834 |
| therapy_p0_t0 | 463 | 505 | 514 |
| therapy_p0_t1 | 323 | 374 | 1024 |

Every reply that hit the 1024 cap was Yoruba. `yo/therapy_p0_t1` hit it on all 7 of its
replies. Yoruba replies use about twice as many tokens as English ones. The likely reason
is tokenizer cost rather than longer content, but langid and a manual read are needed to
confirm this.

## Throughput per turn

| Turn | Active | New tokens | Seconds | tok/s | Hit cap | Conversations written |
|---|---|---|---|---|---|---|
| 1 | 18 | 9315 | 45.2 | 206.2 | 1 | 0 |
| 2 | 18 | 8199 | 43.1 | 190.4 | 1 | 0 |
| 3 | 18 | 8422 | 45.0 | 187.3 | 1 | 0 |
| 4 | 18 | 9754 | 48.2 | 202.5 | 2 | 0 |
| 5 | 18 | 10488 | 51.2 | 204.9 | 1 | 0 |
| 6 | 18 | 9805 | 71.4 | 137.4 | 2 | 0 |
| 7 | 18 | 9552 | 82.0 | 116.5 | 2 | 0 |
| 8 | 17 | 6301 | 67.0 | 94.1 | 1 | 1 |
| 9 | 15 | 7040 | 60.5 | 116.3 | 1 | 3 |
| 10 | 14 | 4903 | 59.5 | 82.4 | 0 | 6 |
| 11 | 11 | 3749 | 35.3 | 106.2 | 0 | 7 |
| 12 | 11 | 3652 | 37.4 | 97.5 | 0 | 7 |
| 13 | 11 | 3875 | 44.9 | 86.3 | 0 | 7 |
| 14 | 9 | 2801 | 35.2 | 79.5 | 0 | 9 |
| 15 | 9 | 3671 | 30.4 | 120.9 | 0 | 18 |

Throughput was about 80–206 tok/s. It dropped from turn 6 onward as contexts grew and the
batch got smaller. With only 18 conversations, the batch is too small to use the GPU fully.
Turn 1 of the full replay reached 1247 tok/s with 540 conversations active.

## Full Gemma replay (status)

The full replay (`/workspace/outputs/replay/`, 540 conversation-language pairs) started
after the pilot and was stopped by a `KeyboardInterrupt` at 18:30 UTC, after turn 1 (259,391
new tokens in 208 s, 1247 tok/s, 40 replies at the cap). Its checkpoint is saved, so it can
be continued with `--resume`.

## Implications

- With the 8192-token context limit, Gemma cannot finish most 15-turn Yoruba conversations,
  so Yoruba's late turns will be missing. Turn-level comparisons across languages past about
  turn 8 will be biased toward English and Hausa. Options: a smaller `--max-new-tokens` for
  Gemma, or analysis restricted to turns every language reaches.
- Still to run: `project.py --check`, `project.py`, `langid.py` and `analyze.py` on
  `gemma_uncapped.jsonl`, plus a manual read of some Hausa and Yoruba replies.
