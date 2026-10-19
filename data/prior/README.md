# Prior conversations (from the fig_ood_drift project)

Source: earlier project by the author, February 2026. Multi-turn conversations in which
Claude Sonnet 4 (via OpenRouter) played a simulated user and Gemma 2 27B (it) answered.
User turns were later translated into Hausa (ha) and Yoruba (yo) and checked by native
speakers. Assistant turns in the translated files are the ORIGINAL ENGLISH replies and
are not used: this project regenerates every reply (replay design, see CLAUDE.md).

## Audit (19 Oct 2026, `scripts/audit_prior.py`)

| Domain | en | ha | yo | User turns per conversation (min / median / max) |
|---|---|---|---|---|
| coding | 60 | 60 | 60 | 8 / 15 / 15 |
| metacognitive | 60 | 60 | 60 | 15 / 15 / 15 |
| therapy | 60 | 60 | 60 | 7 / 15 / 15 |

- Every ha and yo file pairs with an English original (same persona_id, topic_id and
  timestamp), with the same number of user turns.
- No encoding damage: Hausa hooked letters (ɓ ɗ ƙ) and Yoruba dotted letters (ẹ ọ ṣ) are
  stored correctly. Garbled letters seen earlier were a PowerShell display issue only.
- 16 user turns (of about 2,600) were left in English, untranslated: 7 in ha coding,
  2 in yo coding, 2 in yo metacognitive, 5 in yo therapy. These must be translated
  (and checked) or their conversations excluded before the replay; see
  `untranslated_turns.csv`.
- Yoruba tone marks are used only partly (dots under letters are consistent). This is a
  known limitation of the translations, to be stated in the report.
- All conversations were originally generated against Gemma's English replies, so for
  Qwen (and for every language) the replayed user turns do not react to the new replies.
  This is the same for all languages, so cross-language comparisons stay fair.
