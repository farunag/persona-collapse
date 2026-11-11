# Native-speaker spot-check: both rounds

Round 1: 40 far-role replies (asked to play a character). Round 2: 40 default replies (normal assistant).
20 per language in each round, Qwen 3 32B and Gemma 2 27B, judge labels hidden from annotators.
Fluency is 1 (not understandable) to 5 (native-like); annotators leave it empty when the reply is not
in the user's language. Raw ratings: `ratings/human/`. Script: `scripts/score_spotcheck_v2.py`.

## Round 1: far-role replies

| Model | Language | Replies | Rated in the user's language | Fluency ≤ 2 (in-language ratings) | Mean fluency | Understood: yes / partly / no | Judge: addresses user yes | Mean axis score |
|---|---|---|---|---|---|---|---|---|
| Qwen 3 32B | ha | 13 | 45/52 (87%) | 45/45 (100%) | 1.1 | 2 / 8 / 42 | 0/13 (0%) | -0.08 |
| Qwen 3 32B | yo | 12 | 20/24 (83%) | 20/20 (100%) | 1.1 | 4 / 2 / 18 | 1/12 (8%) | +0.03 |
| Gemma 2 27B | ha | 7 | 12/28 (43%) | 0/12 (0%) | 3.6 | 14 / 10 / 4 | 5/7 (71%) | -1.56 |
| Gemma 2 27B | yo | 8 | 10/16 (62%) | 10/10 (100%) | 1.4 | 1 / 6 / 9 | 2/8 (25%) | -2.78 |

Counts in the understood column are ratings (each reply is rated by every annotator).

Agreement between annotators (exact, over every pair rating the same reply):

- ha: role 50/60 (83%); understood 87/120 (72%); reply language 114/120 (95%); fluency 97/120 (81%)
- yo: role 16/20 (80%); understood 16/20 (80%); reply language 20/20 (100%); fluency 18/20 (90%)

Judge role label vs annotator role label: 68/100 (68%).

Judge said the reply addresses the user in 3/73 (4%) of the ratings where an annotator said it did not.

## Round 2: default replies

| Model | Language | Replies | Rated in the user's language | Fluency ≤ 2 (in-language ratings) | Mean fluency | Understood: yes / partly / no | Judge: addresses user yes | Mean axis score |
|---|---|---|---|---|---|---|---|---|
| Qwen 3 32B | ha | 12 | 22/24 (92%) | 20/22 (91%) | 1.6 | 1 / 12 / 11 | 1/12 (8%) | +0.37 |
| Qwen 3 32B | yo | 12 | 22/24 (92%) | 22/22 (100%) | 1.2 | 1 / 12 / 11 | 2/12 (17%) | +0.29 |
| Gemma 2 27B | ha | 8 | 14/16 (88%) | 2/14 (14%) | 3.1 | 7 / 9 / 0 | 7/8 (88%) | -2.34 |
| Gemma 2 27B | yo | 8 | 10/16 (62%) | 9/10 (90%) | 1.9 | 2 / 11 / 3 | 3/8 (38%) | -1.87 |

Counts in the understood column are ratings (each reply is rated by every annotator).

Agreement between annotators (exact, over every pair rating the same reply):

- ha: role 20/20 (100%); understood 17/20 (85%); reply language 20/20 (100%); fluency 13/20 (65%)
- yo: role 20/20 (100%); understood 11/20 (55%); reply language 18/20 (90%); fluency 16/20 (80%)

Judge said the reply addresses the user in 0/25 (0%) of the ratings where an annotator said it did not.

## Reference: mean axis score of all sensitivity replies

| Model | Language | Default | Far-role | Gap |
|---|---|---|---|---|
| Qwen 3 32B | en | +0.49 | -0.89 | 1.37 |
| Qwen 3 32B | ha | +0.31 | -0.09 | 0.40 |
| Qwen 3 32B | yo | +0.30 | -0.10 | 0.41 |
| Gemma 2 27B | en | +0.95 | -0.38 | 1.33 |
| Gemma 2 27B | ha | -2.48 | -1.54 | -0.94 |
| Gemma 2 27B | yo | -2.56 | -1.32 | -1.24 |
