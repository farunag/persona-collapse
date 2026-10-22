# Gate C: the public dataset reproduces the headline

Run 2026-10-22 on Google Colab (CPU), `notebooks/gate_c.ipynb`, from the public dataset
`Faruna01/persona-collapse-runs` with no token. **Verdict: PASS.**

| Check | Result |
|---|---|
| 1. Rerun `scripts/analyze.py` on the raw run files downloaded from the dataset | 24 of 24 rows of `stage2_domain_drift.csv` match; max abs difference 0.00e+00 |
| 2. Recompute every mean score from the flat `turns` table (`load_dataset(..., "turns")`, 15,268 rows) | 24 of 24 cells match, same conversation counts; max abs difference 4.44e-16 |

The first attempt at check 2 failed: `capping_experiment` was null in every row of the uncapped
files, so `datasets` inferred type null and could not read the capped file. Fixed in
`scripts/export_dataset.py` (commit 7c961dc, writes `"none"`), tables re-uploaded, check rerun.
