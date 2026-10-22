# Persona Collapse: An English-Calibrated Persona Monitor Is Partly Blind in Hausa and Yoruba

**Godwin Faruna**
Draft, 22 October 2026. Code: github.com/farunag/persona-collapse ·
Data: huggingface.co/datasets/Faruna01/persona-collapse-runs ·
Workbench: huggingface.co/spaces/Faruna01/persona-collapse

> Draft status: all numbers are from the committed results in `results/`. The Hausa and Yoruba
> role-adoption labels come from an LLM judge and await native-speaker review (Section 5.5).
> References marked [verify] need checking before submission.

## Abstract

Lu et al. (2026) find an "Assistant Axis" in the activations of chat models. Projecting onto
it tracks how far a model has moved from its default helpful-assistant persona, and capping
activations along it stabilises the persona. The axis and the capping setting were derived
from English data. We ask whether they work when the conversation is in Hausa or Yoruba, two
major West African languages with tens of millions of speakers each.

We replay 180 multi-turn conversations per language (coding, therapy, and "metacognitive"
conversations that push the model to reflect on its own mind) through Qwen3-32B and Gemma 2
27B in English, Hausa and Yoruba. We generate fresh replies and project every reply onto the
published axes.

Uncapped, Qwen drifts strongly from its English coding baseline in self-reflection (−0.89 in
normalised units) and therapy (−0.37), but only by −0.13 to −0.20 in Hausa and Yoruba. A
single-turn sensitivity check shows why this does not mean Qwen is steadier in those languages:

- The separation between default replies and role-play replies shrinks from 1.37 in English
  to about 0.40 in Hausa and Yoruba.
- An LLM judge finds Qwen still fully adopts the requested character in 34–54% of Hausa and
  Yoruba replies. When it does so in the user's language, the axis scores those replies like
  ordinary assistant replies (−0.06 vs +0.03 in Hausa; +0.07 vs +0.08 in Yoruba).
- The published activation cap raises English scores by about 0.6 but Hausa and Yoruba scores
  by only about 0.1, consistent with an instrument that barely sees persona in those languages.

For Gemma, raw Hausa and Yoruba scores are dominated by which language the model replies in.

We conclude that English-calibrated persona monitors, and interventions built on them, should
not be assumed to transfer to lower-resource languages. We release all transcripts,
projections, a reproducibility notebook and an interactive workbench.

## 1. Introduction

Chat models are trained to adopt one character: a helpful, honest assistant. Lu et al. (2026)
show that this character corresponds to a direction in activation space, the Assistant Axis.
Projections onto it fall as models are pushed into role-play or emotionally charged,
self-reflective conversations. Clamping ("capping") the projection from below keeps the model
closer to its assistant persona and reduces harmful outputs in their case studies.

This makes the axis attractive as a monitor: a cheap, model-internal signal that a
conversation is pulling the model away from its intended character.

Monitors are only as good as the conditions they were validated in. The axis is computed from
model responses to English prompts across 275 roles, and the capping threshold is the 25th
percentile of English default-persona activations. Hundreds of millions of people talk to chat
models in other languages. Earlier work shows that safety behaviour degrades in lower-resource
languages: refusals are easier to bypass (Yong et al., 2023; Deng et al., 2024) and
intermediate representations of multilingual models are English-centred (Wendler et al.,
2024). We ask three questions:

1. Does the Assistant persona drift more, sooner or differently when the same conversation
   happens in Hausa or Yoruba?
2. Does the English-calibrated cap still stabilise the model in those languages?
3. When the projection changes less in Hausa or Yoruba, is that because the model drifts less,
   or because the axis sees less?

The answer to the third question changes how the first two must be read. Our main
contribution is evidence that, for Qwen3-32B, the axis is a much weaker instrument in Hausa and
Yoruba. Our second contribution is an open dataset and workbench that let anyone inspect the
conversations turn by turn.

## 2. Background and related work

**The Assistant Axis.** Lu et al. (2026) compute, per model and layer, the difference between
the mean activation of default-assistant responses and the mean of 275 role vectors. Per-turn
projection is the mean post-MLP residual activation over a response's tokens at a target layer
(Qwen3-32B: layer 32; Gemma 2 27B: layer 22). Capping applies
h ← h − v · min(⟨h, v⟩ − τ, 0) over a band of layers (Qwen: layers 46–53, τ at the 25th
percentile). We use their released axes, role vectors and capping configuration without
modification, through their `assistant_axis` library.

**Persona and steering vectors.** The axis belongs to a family of linear representations used
for monitoring and control: activation addition (Turner et al., 2023), representation
engineering (Zou et al., 2023) and persona vectors (Chen et al., 2025). Like the axis, these
are usually extracted and validated in English.

**Multilingual safety.** Translating harmful prompts into low-resource languages bypasses
safety training (Yong et al., 2023; Deng et al., 2024). Our own prior work finds that refusal
representations drift in African languages (Faruna, 2026). We extend this concern from
refusals to persona monitoring and stabilisation.

## 3. Data

**Conversations.** We reuse 180 English conversations from an earlier project: 60 each of
coding, therapy and metacognitive. They were written by Claude Sonnet 4 playing a user against
Gemma 2 27B, spread over several personas and topics per domain, with 7–15 user turns each.
- We keep only the user turns.
- The user turns were translated into Hausa and Yoruba and checked by native speakers. Sixteen
  missing turns were translated and checked for this study.
- Conversations are paired across languages by persona and topic, so each language replays
  exactly the same conversations.

**Sensitivity prompts.** For the single-turn check we take 12 first user turns (4 per domain)
in each language. Each is combined with 16 conditions, with the English role instruction
prepended to the user turn:
- 3 default prompts: none, "You are an AI assistant.", "You are a large language model.";
- 3 near roles: analyst, consultant, evaluator;
- 10 far roles: ghost, hermit, bard, leviathan, wraith, trickster, bohemian, prophet, oracle,
  wanderer.

This gives 576 prompts per model.

## 4. Methods

### 4.1 Replay

We feed each conversation's user turns to the model one at a time and generate a fresh reply
after each, carrying the model's own previous replies in the context. There is no system prompt
and no instruction about reply language or length.

| | Qwen3-32B | Gemma 2 27B |
|---|---|---|
| Revision | 9216db57 | aaf20e6b |
| Thinking mode | off | – |
| Uncapped generation | vLLM 0.32, fp8 KV cache, Triton attention | vLLM |
| Capped generation | HF transformers + `build_capping_steerer`, bf16 | – (no published cap) |
| Context limit | 32,768 tokens | 8,192 tokens |
| Sampling | T = 0.7, top-p 0.9, ≤ 1,024 new tokens, seed 0 | same |

- Uncapped: all 540 conversations (180 × 3 languages) for both models.
- Capped: Qwen only, 10 therapy and 10 metacognitive conversations per language, chosen evenly
  spaced across personas and topics before any results were seen.
- Gemma conversations end when the next turn no longer fits in its context: 198 of 540, 136 of
  them in Yoruba.
- The fp8 KV cache affects only sampling. All projections use the bf16 model.

### 4.2 Projection and normalised score

Each finished conversation is passed once through the bf16 model. For every assistant turn we
average the post-MLP residual activations over that turn's response tokens at the target layer
and project onto the published axis, as in Lu et al. Batched and single-conversation
projections agree to within 0.002 raw units.

To compare models we report a normalised score:

score = (projection − μ_roles) / (projection_default − μ_roles),

where μ_roles is the mean projection of the 275 published role vectors. A score of 1 is the
default assistant; 0 is the average role. The reference values are:
- Qwen: default −4.54, role mean −27.14.
- Gemma: default 9,587.7, role mean 8,795.0. Gemma's axis at layer 22 is dominated by hidden
  dimension 4140 (weight 0.57).

### 4.3 Drift and comparable turns

Language alone shifts the projection, so we report **drift**: a conversation's mean score minus
the mean score of that model and language's 60 coding conversations, its own baseline. Because
Hausa and Yoruba use more tokens and Gemma runs out of context, we compare only a
**comparable-turn window** per model and domain. This is the longest run of turns from turn 1
that every language reaches in at least 10 conversations and at least 80% of its turn-1
conversations:

| Condition | Coding | Therapy | Metacognitive |
|---|---|---|---|
| Qwen uncapped | 1–12 | 1–13 | 1–15 |
| Qwen capped | – | 1–10 | 1–15 |
| Gemma uncapped | 1–8 | 1–7 | 1–8 |

Confidence intervals are 95% percentile bootstraps over conversations (2,000 resamples, seed 0).

### 4.4 Reply language

We detect the language of every reply with GlotLID (Kargaran et al., 2023) and flag replies
that are mostly code.

### 4.5 Axis sensitivity and role adoption

For the 576 sensitivity prompts we generate one reply each (≤ 300 tokens) and compute its score.
**Separation** is the mean default score minus the mean far-role score, with a bootstrap CI over
the 12 messages.

To separate "the axis sees less" from "the model role-plays less", `gemini-3.5-flash-lite`
judges every default and far reply (936 replies) at temperature 0, using the paper's rubric:
- **full:** speaks as the role and does not say it is an AI;
- **somewhat:** says it is an AI but clearly shows the role;
- **none:** ignores or refuses the role.

The judge also reports the reply's language and whether the reply addresses the user. A
40-reply stratified sample (20 Hausa, 20 Yoruba, judge labels hidden) is out for native-speaker
rating.

## 5. Results

### 5.1 Reproduction (Gate A)

Before the main runs we checked our pipeline against the paper on its released transcripts.
- **Qwen's capping case studies:** the capped transcripts sit closer to the default than the
  unsteered ones in all three scenarios.

  | Scenario | Capped score | Unsteered score |
  |---|---|---|
  | Jailbreak | 0.57 | −0.40 |
  | Self-harm | 0.43 | −0.58 |
  | Delusion | −0.01 | −1.01 |

- **Domain order:** Qwen reproduces the paper's ordering (coding 0.42, writing 0.36 > therapy
  0.00 > philosophy −0.73).
- **Gemma:** inconclusive, with one transcript per domain, partly from other models. Its domain
  means differ by less than single turns swing.

### 5.2 Uncapped drift by language

**Table 1. Drift from each language's coding baseline over comparable turns (95% CI).**

| Model | Domain | English | Hausa | Yoruba |
|---|---|---|---|---|
| Qwen3-32B | therapy | −0.37 [−0.46, −0.28] | −0.13 [−0.16, −0.11] | −0.18 [−0.20, −0.16] |
| Qwen3-32B | metacognitive | −0.89 [−0.96, −0.83] | −0.14 [−0.16, −0.11] | −0.20 [−0.22, −0.17] |
| Gemma 2 27B | therapy | −0.12 [−0.22, −0.03] | −0.91 [−1.11, −0.74] | +0.02 [−0.30, +0.33] |
| Gemma 2 27B | metacognitive | −0.31 [−0.43, −0.20] | −0.70 [−0.93, −0.46] | −0.40 [−0.61, −0.20] |

Coding baselines:

| Model | English | Hausa | Yoruba |
|---|---|---|---|
| Qwen | 0.34 | 0.27 | 0.30 |
| Gemma | 0.66 | −2.03 | −1.94 |

**Qwen.** Drift in English follows the paper's pattern: self-reflection pulls hardest, therapy
less. In Hausa and Yoruba the drift has the same sign but is 3–6× smaller. The gap from English
is +0.75 [+0.69, +0.82] (Hausa) and +0.70 [+0.63, +0.76] (Yoruba) in metacognitive, and
+0.24 / +0.19 in therapy. Read naively, Qwen would seem to hold its persona much better in
Hausa and Yoruba. Sections 5.4–5.5 show that reading is not supported.

**Gemma.** The coding baselines differ by 2.7 units between English and Hausa or Yoruba, larger
than any within-language drift. Its Hausa and Yoruba drift is large but noisy, and partly
reflects reply language (Section 5.3). We do not interpret it as persona drift.

Figures: `results/stage2_drift_Qwen3-32B_uncapped.png`,
`results/stage2_drift_gemma-2-27b-it_uncapped.png` (per-turn curves).

### 5.3 Reply language and length

- **Qwen** almost always answers in the user's language: 95–99% in Hausa and Yoruba
  therapy and metacognitive. Hausa coding is lower (85%): GlotLID labels 115 of its 817
  replies as a language other than Hausa, English or Yoruba, which we have not yet inspected.
- **Gemma** answers Yoruba therapy users in Yoruba only 77% of the time. Most of the rest are
  in English (134 of 603 replies).
- **Reply length.** Qwen's Hausa and Yoruba replies are long. 72–98% hit the 1,024-token cap,
  against 29–54% in English. Gemma's replies are shorter (218–680 tokens on average), but its
  Yoruba conversations exhaust the 8,192-token context: 43–50 of 60 per domain stop early.
- **Fluency.** Informal reading of the pilot suggests many of Qwen's Hausa replies are fluent
  in form but miss the user's point (for example, a grammar lesson in reply to an anxiety
  question). Native-speaker ratings of comprehension are planned.

### 5.4 The axis is less sensitive in Hausa and Yoruba

**Table 2. Single-turn sensitivity check.** Separation = default − far (95% CI over 12 messages).

| Model | Language | Default | Near | Far | Separation | vs English | In user's language |
|---|---|---|---|---|---|---|---|
| Qwen3-32B | en | 0.49 | 0.48 | −0.89 | 1.37 [1.16, 1.57] | 1.00 | 99% |
| Qwen3-32B | ha | 0.31 | 0.37 | −0.09 | 0.40 [0.33, 0.47] | 0.29 | 86% |
| Qwen3-32B | yo | 0.30 | 0.44 | −0.10 | 0.41 [0.33, 0.48] | 0.30 | 81% |
| Gemma 2 27B | en | 0.95 | 0.93 | −0.38 | 1.33 [1.10, 1.57] | 1.00 | 99% |
| Gemma 2 27B | ha | −2.48 | −1.56 | −1.54 | −0.94 [−1.67, −0.24] | −0.71 | 50% |
| Gemma 2 27B | yo | −2.56 | −0.32 | −1.32 | −1.24 [−2.00, −0.39] | −0.93 | 31% |

- **Qwen.** The same far-role instructions move the score less than a third as far in Hausa
  and Yoruba as in English.
- **Gemma.** Separation has the wrong sign in Hausa and Yoruba. Far-role prompts there are
  mostly answered in English, which raises the score by 3–4 units and outweighs any persona
  effect.

### 5.5 Role adoption: the model still role-plays, the axis does not see it

A smaller separation could mean the model ignores role instructions more often in Hausa and
Yoruba (behaviour), or that it follows them but the axis does not register it (measurement).
The LLM judge lets us separate the two.

**Table 3. Role adoption and scores, far roles (120 replies per cell).**

| Model | Lang | Full | Somewhat | None | Default score | Far judged full |
|---|---|---|---|---|---|---|
| Qwen3-32B | en | 96% | 4% | 0% | 0.49 | −0.88 [−0.97, −0.79] |
| Qwen3-32B | ha | 34% | 18% | 48% | 0.31 | −0.35 [−0.51, −0.20] |
| Qwen3-32B | yo | 54% | 12% | 34% | 0.30 | −0.27 [−0.42, −0.14] |
| Gemma 2 27B | en | 92% | 7% | 1% | 0.95 | −0.44 [−0.55, −0.34] |
| Gemma 2 27B | ha | 89% | 8% | 3% | −2.48 | −1.49 [−1.81, −1.20] |
| Gemma 2 27B | yo | 92% | 5% | 3% | −2.56 | −1.11 [−1.36, −0.89] |

**Table 4. Qwen far-role scores split by reply language** (mean score, n; unpaired, no CI yet).

| Language | Group | Reply in user's language | Reply in another language |
|---|---|---|---|
| Hausa | default | +0.31 (35) | +0.49 (1) |
| Hausa | far, judged full | **−0.06 (25)** | −0.81 (16) |
| Hausa | far, judged none | +0.03 (54) | +0.36 (3) |
| Yoruba | default | +0.27 (33) | +0.60 (3) |
| Yoruba | far, judged full | **+0.07 (41)** | −0.85 (24) |
| Yoruba | far, judged none | +0.08 (41) | – |

For Qwen, both effects are present, and measurement is the larger.
- **Behaviour.** Qwen adopts far roles less often in Hausa (34% full) and Yoruba (54%) than in
  English (96%).
- **Measurement.** When Qwen does fully adopt a role **in Hausa or Yoruba**, the axis scores
  the reply almost exactly like a reply that ignored the role.
  - Default to far-full drops by only 0.37 (Hausa) and 0.20 (Yoruba), 15–27% of the English
    separation.
  - The far-full replies that score low (−0.81, −0.85) are those where the model switched to
    another language, mostly English.

For Gemma, behaviour is ruled out: it adopts far roles about 90% of the time in every language.
Its raw scores are driven by reply language. Among replies in the user's language the
default − far gap has the right sign but is small: 0.85 (Hausa) and 0.33 (Yoruba), against 1.39
in English.

**Caveat.** The Hausa and Yoruba labels come from an LLM judge, which may be lenient on weak or
partly incoherent text. They are provisional until the native-speaker sample is rated.

### 5.6 Capping

**Table 5. Capped minus uncapped, matched conversations (Qwen3-32B).** Same 10 conversations
per domain and language, same turns, mean per-conversation difference with 95% bootstrap CI.

| Domain | Language | Uncapped | Capped | Capped − uncapped |
|---|---|---|---|---|
| therapy (turns 1–10) | en | −0.13 | 0.48 | +0.61 [+0.52, +0.69] |
| therapy | ha | 0.16 | 0.28 | +0.11 [+0.06, +0.19] |
| therapy | yo | 0.13 | 0.22 | +0.09 [+0.06, +0.12] |
| metacognitive (turns 1–15) | en | −0.51 | 0.05 | +0.56 [+0.43, +0.72] |
| metacognitive | ha | 0.13 | 0.23 | +0.10 [+0.06, +0.15] |
| metacognitive | yo | 0.12 | 0.24 | +0.12 [+0.09, +0.15] |

- **English.** The cap lifts capped therapy above the uncapped English coding baseline (0.48
  vs 0.34).
- **Hausa and Yoruba.** The effect is about one sixth as large.
- **Two readings.** The cap may change Hausa and Yoruba generation less, or the axis may simply
  not register the change. Section 5.5 makes the second at least partly true, so this result
  does not by itself show that capping does less in those languages.
- **Side effects.** Capped replies were 100% in the user's language (uncapped 95–100% in these
  domains) and shorter in English: 9–22% hit the token limit, against 29–38% uncapped.

Figure: `results/stage2_capping_Qwen3-32B.png`.

## 6. Discussion

**A monitor's silence is not evidence of safety.**
- Qwen's projection barely moves in Hausa and Yoruba, whether the conversation is a coding
  question or an existential conversation about the model's mind. A dashboard built on this
  axis would show a stable assistant.
- Our sensitivity and judge results show it would show the same thing for a reply that fully
  inhabits a "wraith" or "prophet" character.
- The users least well served by English-centric safety work would get the least protection
  from this monitor.

**A rough sense of scale.** Dividing Qwen's Hausa and Yoruba drift by their relative
sensitivity (0.29–0.30) gives about −0.45 to −0.63 in English-equivalent units. That is larger
than English therapy drift but smaller than English self-reflection drift. This rescaling
assumes sensitivity is uniform along the axis, which we have not shown. We report it only to
indicate that "less drift in Hausa and Yoruba" is not a safe conclusion.

**Why might the axis see less?**
- One possibility is that persona is represented differently once the model processes Hausa
  or Yoruba text. Multilingual models route much of their computation through English-like
  representations (Wendler et al., 2024), and in lower-resource languages more of the
  residual stream may be spent on the language itself.
- Another is that the axis, computed from English responses, captures English stylistic
  correlates of persona (register, vocabulary) that have no counterpart in Hausa or Yoruba
  text.
- Testing these needs axes computed from Hausa and Yoruba role responses, which our pipeline
  could produce with the paper's role prompts translated.

**Practical recommendations.**
1. Validate persona monitors per language before deployment, with a separation check like
   Table 2.
2. Report drift relative to a per-language baseline, never raw.
3. Watch reply language as its own signal. In Gemma, a language switch moves the projection
   more than persona does.
4. Treat a stable projection in a low-resource language as "unknown", not "safe".

## 7. Limitations

- **One sample per conversation (seed 0).** Variance comes from 60 conversations per domain,
  not repeated samples.
- **The capped run is small:** 10 conversations per domain and language, therapy and
  metacognitive only.
- **Long-reply truncation.** Most Qwen Hausa and Yoruba replies hit the 1,024-token limit, so
  projections average over truncated replies.
- **Translated conversations.** User turns were written in English and translated, and may
  read less naturally than native conversations.
- **LLM-judged role adoption** in Hausa and Yoruba is provisional.
- **Sensitivity check scale.** It uses 12 messages per language, English role instructions,
  and replies of at most 300 tokens.
- **Gemma's axis** is dominated by one hidden dimension, and its Gemma Gate A check was
  inconclusive.
- **Two models and two non-English languages.** Results may not generalise to other models,
  languages or to the paper's Llama 3.3 70B.

## 8. Reproducibility and release

- **Raw outputs on Hugging Face:** every transcript with full settings, per-turn projections at
  every 4th layer, language detection, logs and checkpoints.
- **Flat tables:** `turns` (15,268 replies) and `sensitivity` (1,152 replies).
- **Gate C:** `notebooks/gate_c.ipynb` reruns the analysis from the public dataset on a CPU and
  recomputes every mean from the flat table. Both matched the committed results exactly
  (`results/gate_c.md`).
- **Environment records** for both GPU machines are in `env/`.
- **Workbench:** the Hugging Face Space shows each conversation's replies next to its
  per-turn trace, default and capped side by side, in each language.

## Ethics

- **Sensitive content.** The therapy conversations and some role prompts involve distress and
  dark characters. The dataset and workbench carry content warnings and say they are not
  support services.
- **People.** The user turns are synthetic; no real users' conversations were used.
- **Translators.** Native speakers checked the translations and are rating the samples.

## Acknowledgements

This work was funded by the Cosmos Institute (Human Autonomy track). It builds directly on the
code, axes and capping configurations released by Lu et al. (2026). We thank the native-speaker
reviewers of the Hausa and Yoruba translations.

## References

- Chen, R., et al. (2025). Persona vectors: Monitoring and controlling character traits in
  language models. arXiv:2507.21509. [verify]
- Deng, Y., et al. (2024). Multilingual jailbreak challenges in large language models. ICLR.
  arXiv:2310.06474. [verify]
- Faruna, G. (2026). Refusal centroid drift in frontier models. arXiv:2603.19273. [verify title]
- Gemma Team (2024). Gemma 2: Improving open language models at a practical size.
  arXiv:2408.00118. [verify]
- Kargaran, A. H., et al. (2023). GlotLID: Language identification for low-resource languages.
  Findings of EMNLP. arXiv:2310.16248. [verify]
- Lu, C., et al. (2026). The Assistant Axis: Situating and stabilizing the default persona of
  language models. arXiv:2601.10387.
- Qwen Team (2025). Qwen3 technical report. arXiv:2505.09388. [verify]
- Turner, A., et al. (2023). Steering language models with activation engineering.
  arXiv:2308.10248. [verify]
- Wendler, C., et al. (2024). Do llamas work in English? On the latent language of multilingual
  transformers. ACL. arXiv:2402.10588. [verify]
- Yong, Z.-X., Menghini, C., & Bach, S. H. (2023). Low-resource languages jailbreak GPT-4.
  arXiv:2310.02446. [verify]
- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency.
  arXiv:2310.01405. [verify]
