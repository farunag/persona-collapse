# Data manifest

Every file the project needs, where it lives, and what it is used for. Nothing that is only on a
GPU instance counts as saved. Before destroying an instance, run `scripts/verify_artifacts.py`
(it must print `ALL REQUIRED FILES SAVED`) and `scripts/save_env.sh`.

Storage
- **GitHub** (`farunag/persona-collapse`): code, inputs, small results, environment records.
- **Hugging Face dataset** (`Faruna01/persona-collapse-runs`, private until release): every raw
  output of the GPU runs. Made public, with a dataset card, at release.

## 1. Inputs (replication)
| File | Where | What |
|---|---|---|
| `data/prior/` | GitHub | English originals + native-checked ha/yo user turns (fig_ood_drift) |
| `data/prior/untranslated_turns.csv` | GitHub | the 16 turns translated on 19 Oct |
| `data/source/conversations.jsonl` | GitHub | the 540 (conversation, language) rows replayed |
| `lu-christina/assistant-axis-vectors` | public HF | axes (`qwen-3-32b/`, `gemma-2-27b/`) and Qwen capping config |
| model revisions | in every output row | Qwen3-32B `9216db57…`, gemma-2-27b-it `aaf20e6b…` |

## 2. Raw run outputs (HF dataset; replication, workbench, report)
Per model and condition (`qwen_uncapped`, `qwen_capped`, `gemma_uncapped`), under
`<model>/replay/`:

| File | Content |
|---|---|
| `<run>.jsonl` | full transcripts, per-turn token counts, finish reasons, seeds, settings, versions |
| `<run>.log.jsonl` | per-turn throughput log |
| `<run>.ckpt.done.json` | final checkpoint (proof of completion) |
| `<run>.proj.jsonl` | per-turn projection + normalised score, every-4th-layer projections |
| `<run>.langid.jsonl` | detected reply language per turn |

Also `<model>/pilot/` (pilot runs, 400- and 1024-token) and `<model>/logs/` (run and timing logs).

## 3. Environment records (replication) — GitHub `env/<machine>/`
`nvidia-smi`, GPU name, CUDA/driver, `uv pip freeze` of the assistant-axis env and of the vLLM
env, assistant-axis git commit, repo commit, and the exact run commands (`scripts/save_env.sh`).

## 4. Analysis outputs (technical report) — GitHub `results/`
| File | Use |
|---|---|
| `gate_a_*` | reproduction check against the paper |
| `stage2_turn_scores.csv` | per model x condition x language x domain x turn: score, CI, drift |
| `stage2_domain_drift.csv` | mean drift per domain over comparable turns, CI |
| `stage2_gap_vs_en.csv` | Hausa/Yoruba minus English, CI |
| `stage2_reply_language*.csv` | share of replies in the user's language |
| `stage2_tokens.csv` | tokens per turn, cut-off rate, context stops per language |
| `stage2_*.png` | report figures |
| `stage2_summary.md` | key numbers, comparable-turn windows, provenance |
| `ratings/` | native-speaker ratings of a reply sample (Wed) |

Regenerate everything in `results/` from the HF dataset with `scripts/analyze.py` (see README).

## 5. Workbench data — GitHub `space/`
`space/data/` is built from the HF dataset files with `scripts/build_space_data.py` (one file
per model and topic, so the page loads quickly). The pilot sample stays in
`space/sample_data/`.
