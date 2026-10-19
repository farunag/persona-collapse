# Persona Collapse

Measure Assistant-persona drift across English, Hausa, Yoruba, Igbo, Nigerian Pidgin
and French, building directly on Lu et al. 2026, "The Assistant Axis"
(arXiv 2601.10387, code: github.com/safety-research/assistant-axis, cloned at
/workspace/assistant-axis). Funded by a Cosmos Institute grant (Human Autonomy track).

## Models and published settings
| Model | Target layer | Capping |
|---|---|---|
| Qwen/Qwen3-32B (thinking OFF) | 32 | `layers_46:54-p0.25` |
| google/gemma-2-27b-it | 22 | none published → uncapped only |

Axes come from the Hugging Face dataset `lu-christina/assistant-axis-vectors`
(folders `qwen-3-32b/`, `gemma-2-27b/`). Load them with `assistant_axis.load_axis`.
Gemma 2 has an 8,192-token context: check token length before every Gemma run.

## How to work
- Python environment: the one `uv sync` built in /workspace/assistant-axis. Run code with
  `uv run --project /workspace/assistant-axis python scripts/<file>.py`; add packages with
  `uv add --project /workspace/assistant-axis <pkg>`.
- Hardware: 1x RTX PRO 6000 (96 GB). Load models in bfloat16, never quantized
  (the published axes were computed in full precision).
- Use the `assistant_axis` library (load_model, load_axis, extract_response_activations,
  project, build_capping_steerer). Do not reimplement its math. Read its source and
  `notebooks/project_transcript.ipynb` before writing new projection code.
- Per-turn projection = mean post-MLP residual activation over that turn's response
  tokens at the target layer, projected onto the axis (the paper's method).
- Test every script on 1–2 conversations before a full run. Print timings.
- One model on the GPU at a time. Free memory (del model; torch.cuda.empty_cache()) before switching.
- Save raw outputs as JSONL with: model, language, domain, seed, capped, turn, projection.
- Log seeds, model revision and library versions in every results file.
- Commit after each working step with a clear message; push to GitHub.

## Rules
- Never delete or overwrite files in data/ or results/ without asking me first.
- Large files (activations, weights, caches) live under /workspace, never in git.
- Secrets (HF_TOKEN, API keys) come from environment variables only; never write them to files.
- If a result looks surprising, say so and show the evidence before moving on.

## Repo layout
- data/source/        English source conversations
- data/translated/    one JSON per language, with reviewer sign-off
- scripts/            translate, replay, project, analyze, export_dataset, gate_a
- results/            charts and small summary tables
- space/              Hugging Face Space (workbench)
- paper/              report, Cosmos write-up, blog draft
