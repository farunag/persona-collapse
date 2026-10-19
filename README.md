# Persona Collapse

**Does an AI assistant stay itself in every language?**

Persona Collapse measures *persona drift* (an AI model slipping out of its trained
"helpful assistant" character during long conversations) across seven
languages (English, French, Swahili, Hausa, Yoruba, Nigerian Pidgin and Tamil),
spanning high- to low-resource languages and Latin and non-Latin scripts, and turns the results into a public
workbench where anyone can watch it happen.

It builds directly on **The Assistant Axis** (Lu et al., 2026,
[arXiv:2601.10387](https://arxiv.org/abs/2601.10387),
[code](https://github.com/safety-research/assistant-axis)), using its published
axes and activation-capping settings for Qwen 3 32B and Gemma 2 27B.

Supported by a [Cosmos Institute](https://www.cosmos-institute.org) grant
(AI and Human Autonomy track).

> Status: work in progress (October 2026).

## Layout
| Folder | Contents |
|---|---|
| `scripts/` | setup, translation, replay, projection, analysis, dataset export |
| `prompts/` | task prompts used with Claude Code |
| `data/` | source and translated conversations |
| `results/` | charts and summary tables |
| `space/` | Hugging Face Space (workbench) |
| `paper/` | technical report, write-ups |

## Citation
If you use this work, please also cite Lu et al. (2026).
