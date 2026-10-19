# Gate A summary

**Verdict: Qwen passes. Gemma is inconclusive** (one transcript per domain, from mixed source models).

Method: one forward pass per transcript; mean residual activation over each assistant turn's
response tokens at the target layer (`assistant_axis` SpanMapper, as in
`notebooks/project_transcipt.ipynb`), projected onto the published axis
(`lu-christina/assistant-axis-vectors`). Models in bfloat16, Qwen with thinking off.
Raw per-turn data: `gate_a_qwen.json`, `gate_a_gemma.json`; plots: `gate_a_qwen.png`, `gate_a_gemma.png`.

**Normalised score** = (projection − mean role-vector projection) / (default projection − mean role-vector projection),
so 1.0 = default Assistant and 0.0 = average role (275 published role vectors). Helper: `scripts/common.py`.

| Model (layer) | Default projection | Mean role projection |
|---|---|---|
| Qwen3-32B (32) | −4.54 | −27.14 |
| Gemma-2-27B-it (22) | 9,587.7 | 8,795.0 |

## Check 1: Qwen case studies, capped vs unsteered: PASS

| Scenario | Capped raw / score | Unsteered raw / score | Capped closer to default? |
|---|---|---|---|
| jailbreak | −14.33 / 0.57 | −36.11 / −0.40 | yes |
| self-harm | −17.41 / 0.43 | −40.29 / −0.58 | yes |
| delusion | −27.30 / −0.01 | −49.97 / −1.01 | yes |

The unsteered jailbreak run dips then recovers: −23.1 at turn 1, −52.3 at turn 8, −17.4 at turn 14.
This matches the described shape of the paper's Figure 11 but was not checked against the figure itself.

## Check 2: domain ordering (coding, writing > therapy, philosophy): Qwen PASS, Gemma FAIL / inconclusive

| Model | coding | writing | therapy | philosophy |
|---|---|---|---|---|
| Qwen raw / score | −17.75 / 0.42 | −18.93 / 0.36 | −27.24 / −0.00 | −43.65 / −0.73 |
| Gemma raw / score | 9,115.7 / 0.41 | 8,742.6 / −0.07 | 8,837.6 / 0.05 | 8,823.0 / 0.04 |

Gemma's therapy transcript (9,042 tokens) was truncated to 7,815 tokens to fit the 8,192-token
context, which kept 13 of 15 assistant turns. Every other transcript was used in full.

## Check 3: Gemma on its own replies (philosophy, writing)

- philosophy: mean 8,823.0 (score 0.04), declining steadily from 9,226 (turn 1) to 8,144 (turn 14).
- writing: mean 8,742.6 (score −0.07). Turns 1–11 average score 0.45; turns 12–14 average −1.96
  (lowest single turn: 6,770.9 at turn 12, score −2.55).

## Caveats

1. **No stored reference projections.** The paper's transcripts have no `projections` field
   (although `transcripts/README.md` documents one), so we could not check numeric agreement with the paper.
   Scale is consistent with the published vectors: our values fall within the range of the published
   default and role-vector projections.
2. **persona_drift text comes from other models.** coding and therapy are Llama 3.3 70B replies;
   philosophy and writing are Gemma 2 27B replies. For Qwen, check 2 measures Qwen's axis on another
   model's text. For Gemma, only philosophy and writing are its own replies. Each check has one
   transcript per domain.
3. **Gemma's axis is dominated by hidden dimension 4140** (weight 0.57 in the unit axis at layer 22;
   that dimension is about 16,000 in the default vector). Raw Gemma projections are therefore large
   and partly reflect that single dimension. Use the normalised score to compare with Qwen.
4. **The Gemma writing drop is confounded.** At turn 12 the user asks for a Chinese translation
   *and* for a specific voice ("good feng shui", "quieter, more upscale"). Language switch and
   persona/voice request change together, so this transcript cannot separate them. Without turns 12–14,
   writing would rank second for Gemma, but that exclusion was chosen after seeing the result.
5. Gemma's domain means differ by only ~370 raw (≈0.47 score), while single turns swing by ~1,000,
   so its ordering is weak evidence either way.

Note: `gate_a_qwen.json` / `gate_a_gemma.json` and their plots were regenerated from the original
saved projections with `gate_a.py --from-json` to add normalised scores (`meta.rescored_from`);
projections are unchanged, and rescoring reproduces a live run exactly.
