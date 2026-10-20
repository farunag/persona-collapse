# Stage 2 — Replay pipeline (Tue 20 Oct)

Read CLAUDE.md first. This morning runs on a Windows laptop with NO GPU: write the code,
test everything that can run on CPU, commit and push. The GPU runs happen this afternoon.
Do not reimplement assistant_axis math; reuse scripts/common.py and the gate_a.py approach.

## 1. scripts/build_conversations.py  (runs on laptop)
- Read data/prior/raw/scaled-n60/{coding,therapy,metacognitive}/*.json (English) and
  data/prior/{domain}_translated/{ha,yo}/*.json.
- Take ONLY the user turns. Fill the 16 untranslated turns from
  data/prior/untranslated_turns.csv (match on file + user_turn; read with encoding
  utf-8-sig). Assert afterwards that no ha/yo user turn is identical to its English one.
- conv_id = "{domain}_p{persona_id}_t{topic_id}". Assert en/ha/yo have the same conv_ids
  and the same number of user turns per conversation.
- Write data/source/conversations.jsonl, one row per (conv_id, lang):
  {conv_id, domain, persona_id, topic_id, lang, user_turns: [...], source_file}.
  Expect 540 rows. Print counts per domain x lang and the mean characters per user turn.

## 2. scripts/replay.py  (GPU; write now, run this afternoon)
Arguments: --model {qwen,gemma} --langs en,ha,yo --domains coding,therapy,metacognitive
--limit N (conversations per domain, for pilots) --capped (Qwen only)
--max-new-tokens 400 --temperature 0.7 --top-p 0.9 --seed 0 --out DIR --resume.
- Replay design: for each conversation, turn by turn, append the fixed user turn, generate
  one reply, append it, continue. NO system prompt (as in the paper). Do not tell the
  model which language to answer in: whatever language it chooses is data.
- Qwen: apply_chat_template(..., enable_thinking=False). Gemma: its own chat template.
- Batch across conversations: at turn t, generate replies for every still-active
  conversation in one batch.
- Backends behind one interface:
  - uncapped: vLLM (bfloat16). If vLLM won't install or run on this GPU (RTX PRO 6000,
    Blackwell, CUDA 12.8+), fall back to Hugging Face batched generate (left padding).
  - capped (Qwen only): Hugging Face model + assistant_axis build_capping_steerer with the
    published experiment "layers_46:54-p0.25", batched.
- Context limit: before each turn, if prompt tokens + max_new_tokens would exceed the model
  limit (Gemma 8192; Qwen use 32768), stop that conversation and record stopped_at_turn.
- Output: one JSONL per (model, capped) under --out (default /workspace/outputs/replay),
  one line per finished conversation: conv_id, domain, lang, model, model_revision,
  capped, seed, sampling params, messages, stopped_at_turn, per-turn reply token counts,
  finish_reason per turn, wall time. --resume skips conv_ids already in the file.
- Print progress and throughput (tokens/s) per turn.

## 3. scripts/project.py  (GPU)
- For each finished conversation: one forward pass with the HF model, per-turn mean
  activation over each reply's tokens at the target layer (assistant_axis SpanMapper, as in
  gate_a.py), projection onto the published axis, and the normalised score from
  scripts/common.py. Also store projections at every 4th layer for the workbench heatmap.
- Batch where memory allows; resume support; output JSONL next to the replay file.
- Projection is always measured WITHOUT capping hooks, also for capped transcripts.

## 4. scripts/langid.py  (laptop-testable)
- Detect the language of every reply with GlotLID (Hugging Face cis-lmu/glotlid, fastText).
  Store top label + probability. Map to en / ha / yo / other.

## 5. scripts/analyze.py  (laptop-testable with fake data)
- Per model x lang x domain x turn: mean normalised score with a 95% bootstrap CI over
  conversations.
- Comparable turns only: for each model, use turn positions reached by >= 10
  conversations in EVERY language of that domain.
- Per-language baseline: drift = score minus that language's coding-domain mean.
- Reply-language table: share of replies in the user's language, per lang x domain x turn.
- Charts in results/: drift curves by language (one panel per domain, per model);
  capped vs uncapped (Qwen); reply-language share.
- Write results/stage2_summary.md with the key numbers.

## 6. Tests on the laptop
- Run build_conversations.py for real and commit data/source/conversations.jsonl.
- Unit-test analyze.py and langid.py on a small fake replay/projection file
  (tests/ folder, pytest).
- Dry-run replay.py with a stub generator (--dry-run) that echoes "stub reply" so the
  turn loop, batching, context stop, resume and output format are exercised without a GPU.
- Do NOT download Qwen or Gemma on the laptop.

## 7. Finish
- Commit and push with clear messages. Then give me: row counts from step 1, which tests
  passed, anything you could not test without a GPU, and the exact commands to run on the
  GPU this afternoon in order (setup check, 2-conversation pilot per language for Qwen,
  full Qwen run, projection, langid, analyze).
