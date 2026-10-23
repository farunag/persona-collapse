# Native-speaker spot-check of the role-adoption judge

40 far-role sensitivity replies (20 Hausa, 20 Yoruba; Qwen3-32B and Gemma 2 27B), judge labels hidden.
Two native-speaker annotators per language. Raw ratings: `ratings/human/`. Joined table:
`role_spotcheck_scored.csv` (`scripts/score_spotcheck.py`). Annotators: Austin Abuh, James Joy,
Komolafe Benjamin, Christain Ayo (files are anonymised A/B).

## Reliability

| | Hausa | Yoruba |
|---|---|---|
| Annotator A vs B, role label | 7/20 | 16/20 |
| Annotator A vs B, reply language | 20/20 | 20/20 |
| Judge vs annotator A, role label | 15/20 | 12/20 |
| Judge vs annotator B, role label | (excluded) | 11/20 |

Hausa annotator B rated all 20 replies "full", including replies their own notes describe as
"gibberish/nonsense" and "not understandable"; we read this as a misunderstanding of the label and
exclude their role labels. Their reply-language labels match annotator A on 20/20 and are kept.

## Findings

1. **Qwen's in-language far-role replies are mostly not language.** All 22 Qwen replies written
   in Hausa or Yoruba were rated fluency 1–2 of 5 by every annotator who rated them, and none was
   judged to understand the user ("yes" 0 times; "partly" 1 time).
2. **The judge over-reads these replies.** It labelled some of them "full" or "somewhat" role-play;
   annotators mostly said "none". Both Qwen in-language replies the judge called "full" (Hausa
   leviathan_q09, Yoruba prophet_q02) were rated "somewhat"/"none" with fluency 1–2.
3. **The judge is reliable where the text is fluent.** Every reply in English or mixed language that
   the judge called "full" was rated "full" by the annotators (12 of 12 ratings), and Gemma's
   fluent Hausa role-play (fluency 3–4) was rated "full" as the judge said.
4. **Scores of the broken Qwen replies** sit between −0.22 and +0.26, close to Qwen's
   in-language default replies (+0.31 Hausa, +0.27 Yoruba) and well above Qwen's English role-play
   (about −0.9).

## What changes

- **Not supported:** "Qwen fully role-plays in Hausa/Yoruba and the axis does not see it." The
  sample contains only two Qwen in-language replies the judge called "full", and humans rejected
  both. The judge's ha/yo adoption rates for Qwen (34% / 54% full) are overestimates.
- **Supported:** the axis separates default from far-role prompts much less in Hausa and Yoruba
  (0.40 vs 1.37); it registers role-play when Qwen writes it in English; Gemma's fluent Hausa
  role-play moves its score.
- **New finding:** asked to play a character in Hausa or Yoruba, Qwen mostly degenerates into
  unintelligible, looping text, and the axis scores that text close to its normal in-language
  replies. A monitor built on the axis would not flag that the model has stopped making sense.
- **Open:** whether Qwen's *default* Hausa/Yoruba replies and its long-conversation replies are
  fluent. They were not rated. If they are also degenerate, the in-language default vs far
  comparison is between two kinds of broken text.

## Limitations

40 replies; the sample was stratified by judge label, not by reply language, so only 2 Qwen
in-language "full" replies were rated. A simple repetition metric (repeated trigrams, zlib ratio)
does not track the annotators' fluency ratings well, so we do not extrapolate fluency to all 1,152 replies.
