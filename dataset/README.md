---
license: cc-by-4.0
language:
- en
- ha
- yo
pretty_name: Persona Collapse - multilingual Assistant-persona drift
size_categories:
- 10K<n<100K
task_categories:
- text-generation
tags:
- ai-safety
- interpretability
- activation-steering
- persona
- multilingual
- african-languages
- hausa
- yoruba
configs:
- config_name: turns
  default: true
  data_files:
  - split: train
    path: data/turns-*.jsonl
- config_name: sensitivity
  data_files:
  - split: train
    path: data/sensitivity.jsonl
---

# Persona Collapse: multilingual Assistant-persona drift

**Content warning.** The therapy conversations discuss stress, anxiety, drinking, loneliness and
similar topics. Some sensitivity prompts ask the model to play dark characters (a ghost, a
wraith, a leviathan). This dataset is a research resource, not mental-health support.

The same long conversations, held with two open chat models in **English, Hausa and Yoruba**,
with every reply measured on the **Assistant Axis** of
[Lu et al. 2026](https://arxiv.org/abs/2601.10387): a direction in the model's activations that
separates its default helpful-assistant persona from role-play characters. The question:
*does the assistant persona drift differently, and do English-built safety tools still see it,
when the conversation is not in English?*

- **Code, analysis and reports:** [github.com/farunag/persona-collapse](https://github.com/farunag/persona-collapse)
- **Workbench:** [Hugging Face Space, link to come]
- Built on [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)
  and the published axes in
  [lu-christina/assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors).
- Funded by a [Cosmos Institute](https://cosmos-institute.org) grant (Human Autonomy track).

## Main findings (preliminary)

Native-speaker review of the Hausa and Yoruba judgements is under way. See
`results/stage2_final_report.md` and `results/role_adoption_report.md` in the GitHub repo.

1. **The Assistant Axis is much less sensitive in Hausa and Yoruba.** For Qwen3-32B, the gap
   between default replies and replies in a role-play character is 1.37 (normalised units)
   in English but about 0.40 in Hausa and Yoruba.
2. **Role-play written in Hausa or Yoruba is almost invisible to the axis.** Qwen replies that
   an LLM judge rated as *fully* in character score like normal assistant replies when written
   in Hausa (-0.06 vs +0.03) or Yoruba (+0.07 vs +0.08). The axis only registers the character
   when the model switches to English (-0.81 and -0.85).
3. **Activation capping moves English about six times more than Hausa or Yoruba.** On the
   same 10 conversations per topic and language, the published Qwen cap raised the score by
   +0.61 / +0.56 in English (therapy / self-reflection) and by +0.09 to +0.12 in Hausa and Yoruba.
4. **Gemma 2 27B's Hausa and Yoruba scores are dominated by reply language**, because
   role-play prompts often get English replies. Read its ha/yo drift as a mixture of persona
   and language.

## What is inside

| Config | Rows | One row is |
|---|---|---|
| `turns` | 15,268 | one model reply in a multi-turn conversation |
| `sensitivity` | 1,152 | one single-turn reply to a default or role-play prompt, with an LLM judge's label |

`turns` breaks down as:

| File | Model | Condition | Conversations | Rows |
|---|---|---|---|---|
| `data/turns-qwen_uncapped.jsonl` | Qwen/Qwen3-32B (thinking off) | uncapped | 540 (180 per language) | 7,644 |
| `data/turns-qwen_capped.jsonl` | Qwen/Qwen3-32B | capped, `layers_46:54-p0.25` | 60 (therapy and self-reflection, 10 each per language) | 873 |
| `data/turns-gemma_uncapped.jsonl` | google/gemma-2-27b-it | uncapped | 540 | 6,751 |

Each language has 60 conversations in each of three topics: **coding** (the stable
baseline), **therapy**, and **metacognitive** (pushing the model to reflect on its own mind,
the paper's "philosophy" domain). Conversations are paired across languages by `conv_id`:
the Hausa and Yoruba user turns are translations of the English ones.

The raw run outputs are kept alongside, unchanged: `qwen/replay/`, `gemma/replay/` (full
transcripts with every setting, per-layer projections, language detection, checkpoints),
`*/pilot/`, `*/logs/` and `sensitivity/`.

### `turns` fields

| Field | Meaning |
|---|---|
| `model`, `model_revision` | Hugging Face model id and commit |
| `condition`, `capped`, `capping_experiment` | `uncapped` or `capped`; the capping setting used |
| `language` | language of the user turns: `en`, `ha`, `yo` |
| `domain` | `coding`, `therapy`, `metacognitive` |
| `conv_id`, `persona_id`, `topic_id` | conversation id, shared across languages |
| `seed`, `request_seed` | run seed and the per-reply sampling seed |
| `turn` | turn number, from 1 |
| `user_text`, `reply_text` | the user turn and the model's fresh reply |
| `prompt_tokens`, `reply_tokens`, `finish_reason` | context size before the reply, reply length, `stop` or `length` |
| `projection` | mean activation over the reply's tokens at the target layer, projected onto the Assistant Axis |
| `score` | normalised projection: 1 = the model's default Assistant, 0 = the average role-play character |
| `target_layer`, `layer_ids`, `layer_projections` | target layer (Qwen 32, Gemma 22) and the projection at every 4th layer |
| `reply_language`, `reply_language_label`, `reply_language_prob` | reply language detected by GlotLID |
| `reply_is_code`, `reply_matches_user_language` | reply is mostly code; detected language equals the user's |
| `temperature`, `top_p`, `max_new_tokens`, `backend` | sampling settings and generation backend |

### `sensitivity` fields

`model`, `model_revision`, `language`, `conv_id`, `group` (`default`, `near` or `far`), `role`,
`instruction` (the English role instruction prepended to the user turn), `question_from`
(the conversation the question was taken from), `user_text`, `reply_text`, `reply_tokens`,
`finish_reason`, `projection`, `score`, and the judge's output: `judge`, `role_label`
(`full`, `somewhat`, `none`), `judged_reply_language`, `addresses_user`, `judge_reason`.
Only `default` and `far` replies were judged.

## How it was made

1. **Conversations.** 60 English conversations per topic, written by Claude Sonnet 4 playing a
   user, from an earlier project. Only the user turns were kept. They were translated into
   Hausa and Yoruba and checked by native speakers.
2. **Replay.** For each conversation and language, the user turns were fed to the model one
   by one and the model wrote a fresh reply each time. There was no system prompt and no
   instruction about reply language.
   - Uncapped runs used vLLM with an fp8 KV cache.
   - The capped run used Hugging Face `transformers` with `assistant_axis.build_capping_steerer`.
   - Settings: temperature 0.7, top_p 0.9, up to 1,024 new tokens per reply, seed 0.
   - Gemma stops when its 8,192-token context is full (198 conversations, 136 of them Yoruba).
3. **Projection.** Each finished conversation was run once through the bf16 model.
   - The per-turn projection is the mean post-MLP residual activation over that reply's
     tokens at the target layer, projected onto the published axis, as in the paper.
   - Batched and single-conversation projections agree to within 0.002 raw units.
4. **Reply language.** Detected with [GlotLID](https://huggingface.co/cis-lmu/glotlid).
5. **Sensitivity check.** 12 first user turns per language, each paired with 16 conditions:
   - 3 default prompts;
   - 3 near roles (analyst, consultant, evaluator);
   - 10 far roles (ghost, hermit, bard, leviathan, wraith, trickster, bohemian, prophet,
     oracle, wanderer).

   `gemini-3.5-flash-lite` judged every default and far reply with the paper's role rubric.

## How to use it

```python
from datasets import load_dataset

turns = load_dataset("Faruna01/persona-collapse-runs", "turns", split="train")
sens = load_dataset("Faruna01/persona-collapse-runs", "sensitivity", split="train")

df = turns.to_pandas()
qwen = df[(df.model == "Qwen/Qwen3-32B") & (~df.capped)]
print(qwen.groupby(["language", "domain"]).score.mean().unstack())
```

To compare languages, use **drift**: the score minus the same model and language's
coding-domain mean. Language alone shifts the raw score, most of all for Gemma. Compare
the same turn positions across languages (`scripts/analyze.py` in the repo does both and
reproduces every table and chart).

## Limitations

- **One sample per conversation** (seed 0). The spread comes from the many conversations
  per topic, not from repeated samples.
- **The capped run is small:** 10 conversations per topic and language, therapy and
  self-reflection only, so it has no per-language coding baseline.
- **Many Qwen replies in Hausa and Yoruba hit the 1,024-token limit** (72-98%, against far
  fewer in English). The projection averages over the tokens that were generated.
- **The role judgements for Hausa and Yoruba come from an LLM judge.** Native-speaker review
  of a 40-reply sample is under way and will be added here.
- **The axes and the capping setting were calibrated by their authors in English.** That is
  part of what this dataset tests, but it also means the scores are not a calibrated measure
  of persona in Hausa or Yoruba.
- **Translated user turns** may read less naturally than conversations first written in
  Hausa or Yoruba.

## Licence

- **This dataset** (annotations, scores, translations and the arrangement of the data):
  CC BY 4.0.
- **Model replies** were generated with Qwen3-32B (Apache 2.0) and Gemma 2 27B
  ([Gemma Terms of Use](https://ai.google.dev/gemma/terms)). Use of the Gemma outputs should
  respect Gemma's Prohibited Use Policy.

## Citation

```bibtex
@misc{faruna2026personacollapse,
  title  = {Persona Collapse: Assistant-persona drift in English, Hausa and Yoruba},
  author = {Faruna, Godwin},
  year   = {2026},
  url    = {https://github.com/farunag/persona-collapse}
}
```

Please also cite the Assistant Axis paper:

```bibtex
@misc{lu2026assistantaxis,
  title  = {The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models},
  author = {Lu, Christina and others},
  year   = {2026},
  eprint = {2601.10387},
  archivePrefix = {arXiv}
}
```

Contact: Godwin Faruna ([farunag.github.io](https://farunag.github.io)).
