# Stage 2 run report: projection, reply language and analysis of the uncapped runs

Run 2026-10-21 (UTC) on the Gemma machine (1x NVIDIA A100-SXM4-80GB).
Projection libraries: torch 2.9.0+cu128, transformers 4.57.5, Python 3.11.17, bfloat16.
Models: google/gemma-2-27b-it rev `aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0` (layer 22),
Qwen/Qwen3-32B rev `9216db5781bf21249d130ec9da846c4624c16137` (layer 32). Generation seed 0.
Dataset: `Faruna01/persona-collapse-runs` (private).

## 1. `project.py --check` (batched vs single-conversation projection)

Both checks use the first conversation, `therapy_p5_t0/en` (7 replies).

| Model | Per-turn projection (single = batched) | max \|single − batched\| | In normalised-score units |
|---|---|---|---|
| gemma-2-27b-it | 8592.92, 9089.37, 9086.18, 9917.17, 9293.60, 9338.82, 9192.82 | 0.0020 | 0.0000 |
| Qwen3-32B | 2.46, −10.97, −22.59, −12.14, −18.20, −12.37, −26.80 | 0.0000 | 0.0000 |

Batching does not change the projections. Gemma's raw projections are large (~9,000) because of the
scale of its residual stream. The normalised score corrects for this: Gemma's default projection is
9587.7 and its role-mean projection is 8795.0. Qwen's are −4.54 and −27.14.

## 2. Row counts

| File | Rows |
|---|---|
| `gemma_uncapped.jsonl` (replay) | 540 conversations |
| `gemma_uncapped.proj.jsonl` | 6,751 replies |
| `gemma_uncapped.langid.jsonl` | 6,751 replies (en 2,511 · ha 2,422 · yo 1,818) |
| `qwen_uncapped.jsonl` (replay) | 540 conversations |
| `qwen_uncapped.proj.jsonl` | 7,644 replies |
| `qwen_uncapped.langid.jsonl` | 7,644 replies (en 2,548 · ha 2,548 · yo 2,548) |

No `*.errors.jsonl` was written, so every conversation was projected, and every generated reply has a
projection. Gemma has fewer replies than Qwen, mostly in Yoruba, because many Gemma conversations
stopped at its 8,192-token context. Details are in the token-budget table of `stage2_summary.md`.

Langid (share of replies in the user's language): Gemma en 98.9%, ha 96.2%, yo 89.1%;
Qwen en 97.3%, ha 92.5%, yo 97.8%.

## 3. Upload verification

Each local file was compared with the dataset listing (`list_repo_tree`, recursive), by path and byte size.
**14 of 14 files are present with identical sizes.**

| Dataset path | Bytes |
|---|---|
| gemma/replay/gemma_uncapped.ckpt.done.json | 194 |
| gemma/replay/gemma_uncapped.jsonl | 12,020,698 |
| gemma/replay/gemma_uncapped.langid.jsonl | 2,949,752 |
| gemma/replay/gemma_uncapped.log.jsonl | 2,430 |
| gemma/replay/gemma_uncapped.proj.jsonl | 8,321,018 |
| gemma/pilot/gemma_uncapped.ckpt.done.json | 194 |
| gemma/pilot/gemma_uncapped.jsonl | 428,143 |
| gemma/pilot/gemma_uncapped.log.jsonl | 2,323 |
| gemma/logs/pilot_gemma.log | 26,602 |
| gemma/logs/replay_gemma.log | 27,822 |
| gemma/logs/replay_gemma_resume.log | 40,071 |
| gemma/logs/setup_vllm.log | 9,246 |
| qwen/replay/qwen_uncapped.proj.jsonl | 10,226,969 |
| qwen/replay/qwen_uncapped.langid.jsonl | 3,293,553 |

## 4. Comparable-turn window per model and domain

**Read this first.** The committed `stage2_*` files in `results/` were generated before I pulled
commit 8da91c8. That commit adds `--min-retention` (default 0.8) to `analyze.py`: a turn counts only
if at least 80% of each language's conversations reach it. I re-ran the new `analyze.py` into a
scratch directory and did not overwrite `results/`. The window shrinks a lot for Gemma:

| Model | Domain | Committed results (≥10 conversations) | New analyze.py (≥10 and ≥80% retention) |
|---|---|---|---|
| Qwen3-32B | coding | 1–15 (15) | 1–12 (12) |
| Qwen3-32B | metacognitive | 1–15 (15) | 1–15 (15) |
| Qwen3-32B | therapy | 1–15 (15) | 1–13 (13) |
| gemma-2-27b-it | coding | 1–12 (12) | 1–8 (8) |
| gemma-2-27b-it | metacognitive | 1–15 (15) | 1–8 (8) |
| gemma-2-27b-it | therapy | 1–14 (14) | 1–7 (7) |

For Qwen the conclusions don't change: the gaps vs English stay within ±0.01. For Gemma some do. Gap vs
English, committed → new window: metacognitive/ha **+0.00 → −0.39 [−0.66, −0.13]**,
metacognitive/yo +0.21 → −0.09, therapy/ha −0.56 → −0.79, therapy/yo +0.26 → +0.14.
The committed `results/stage2_*` need regenerating with `--overwrite` before anyone interprets them.
I didn't do it because the project rules say to ask before overwriting anything in `results/`.

## 5. Surprising results (evidence in `stage2_summary.md`)

- **Gemma's coding baseline depends heavily on language:** en +0.72, ha −1.98, yo −1.89. These are
  normalised scores, where 0 is the role mean, so Hausa and Yoruba coding replies project far below
  the average role vector. The per-language baseline handles this, but it is large.
- **Qwen replies in Hausa and Yoruba are mostly cut at max_new_tokens:** coding 97–98%,
  metacognitive 83–93%, therapy 72–89%. English is 29–55%. Projections therefore cover truncated replies.
- Gemma yo/therapy: only 77% of replies are in Yoruba, and 134 are in English.
  Qwen ha/coding: 85% are in Hausa, and 115 are classed "other".

## 6. Problems during the run and how they were fixed

1. **Gemma is gated, and the first `--check` returned a 401.** Loading the model contacted the Hub
   without a token. The models and the Gemma axis were fully cached, so I ran `project.py` with
   `HF_HUB_OFFLINE=1`. `langid.py` downloads GlotLID, so it got the `HF_TOKEN` prefix.
2. **Gemma projection ran out of GPU memory at batch size 1** after 56 conversations, on a long
   conversation (eager-attention softmax; 11.2 GiB reserved but unallocated). I resumed it with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `--resume`, and the remaining 484
   conversations completed. Rows are written per finished conversation, so the resume left no partial
   conversations.
3. **`fasttext` was missing from the assistant-axis env.** I added it with
   `uv add --project /workspace/assistant-axis fasttext-numpy2-wheel`, since numpy is 2.2.6. This
   changes `pyproject.toml` and `uv.lock` in `/workspace/assistant-axis`. Those changes are local and
   not committed, because that repo is the upstream clone.
4. **Every Qwen score was NaN on the first pass.** The Qwen `role_vectors/` were not in the HF cache.
   With `HF_HUB_OFFLINE=1`, the download fallback in `common.axis_dir` returned the cached snapshot
   without error, so `role_mean_projection = mean([])` = NaN. Raw projections were unaffected. I
   downloaded `qwen-3-32b/*` from `lu-christina/assistant-axis-vectors` (275 role vectors), re-ran
   `--check` and the full projection, and replaced the dataset file. The NaN version is kept at
   `/workspace/outputs/stage2_bad/qwen_uncapped.proj.nan_scores.jsonl`. Suggestion: `axis_dir` should
   raise an error when there are no role vectors, rather than returning an empty folder.

Run logs for this session are in `/workspace/outputs/stage2_logs/` on this machine.
They have not been uploaded to the dataset.

## 7. Pre-destroy artifact check (`verify_artifacts.py --machine gemma --upload-missing`)

Run after `bash scripts/save_env.sh gemma-a100` (environment record pushed to `env/gemma-a100/`). Nothing was missing, so nothing was uploaded. Output:

```
      ok  gemma/replay/gemma_uncapped.ckpt.done.json  (0.0 MB)
      ok  gemma/replay/gemma_uncapped.jsonl  (12.0 MB)
      ok  gemma/replay/gemma_uncapped.langid.jsonl  (2.9 MB)
      ok  gemma/replay/gemma_uncapped.log.jsonl  (0.0 MB)
      ok  gemma/replay/gemma_uncapped.proj.jsonl  (8.3 MB)
      ok  qwen/replay/qwen_uncapped.jsonl  (25.1 MB)
      ok  qwen/replay/qwen_uncapped.langid.jsonl  (3.3 MB)
      ok  qwen/replay/qwen_uncapped.proj.jsonl  (10.2 MB)
      ok  gemma/pilot/gemma_uncapped.ckpt.done.json  (0.0 MB)
      ok  gemma/pilot/gemma_uncapped.jsonl  (0.4 MB)
      ok  gemma/pilot/gemma_uncapped.log.jsonl  (0.0 MB)
      ok  gemma/logs/pilot_gemma.log  (0.0 MB)
      ok  gemma/logs/replay_gemma.log  (0.0 MB)
      ok  gemma/logs/replay_gemma_resume.log  (0.0 MB)
      ok  gemma/logs/setup_vllm.log  (0.0 MB)

Required for this machine's runs:
       ok  gemma/replay/gemma_uncapped.jsonl
       ok  gemma/replay/gemma_uncapped.log.jsonl
       ok  gemma/replay/gemma_uncapped.proj.jsonl
       ok  gemma/replay/gemma_uncapped.langid.jsonl

ALL REQUIRED FILES SAVED
```
