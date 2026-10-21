"""Combine role judgments with projections: role-adoption rate per model x language, and the score
of far-role replies by judged label. If replies judged "full" role-play in Hausa/Yoruba score far
below default (as in English), the axis does read persona there and the smaller ha/yo separation
came from less role-play; if they score near default, the axis is less sensitive in that language.
Also writes a spot-check sheet for native speakers (ha/yo far-role replies, stratified by label).

  python scripts/analyze_judgments.py --judgments role_judgments.jsonl \
      --proj /workspace/outputs/sensitivity/{qwen,gemma}_uncapped.proj.jsonl \
      --replay /workspace/outputs/sensitivity/{qwen,gemma}_uncapped.jsonl --out results
"""

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def mean_ci(v, rng, n=2000):
    if not v:
        return None, None, None
    b = sorted(st.mean(rng.choice(v) for _ in v) for _ in range(n))
    return st.mean(v), b[int(.025 * n)], b[int(.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--proj", nargs="+", required=True)
    ap.add_argument("--replay", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--sheet-size", type=int, default=20, help="spot-check rows per language")
    a = ap.parse_args()
    J = read(a.judgments)
    score = {(r["model"], r["conv_id"], r["lang"]): r["score"] for p in a.proj for r in read(p) if r["turn"] == 1}
    reply = {(r["model"], r["conv_id"], r["lang"]): r["messages"] for p in a.replay for r in read(p)}
    rng = random.Random(0)

    rows, L = [], ["# Role adoption (LLM judge) and axis scores", "",
                   f"Judge: {', '.join(sorted({j['judge'] for j in J}))}. Far roles = 10 role-play-end roles; default = 3 neutral prompts.", "",
                   "| Model | Language | Far: full | Far: somewhat | Far: none | Score, default | Score, far judged full (95% CI) | Score, far judged none | n far |",
                   "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted({j["model"] for j in J}):
        for lang in ("en", "ha", "yo"):
            far = [j for j in J if j["model"] == m and j["lang"] == lang and j["group"] == "far"]
            dfl = [score[(m, j["conv_id"], lang)] for j in J if j["model"] == m and j["lang"] == lang and j["group"] == "default"
                   and (m, j["conv_id"], lang) in score]
            if not far:
                continue
            share = {k: sum(j["role_label"] == k for j in far) / len(far) for k in ("full", "somewhat", "none")}
            by = defaultdict(list)
            for j in far:
                s = score.get((m, j["conv_id"], lang))
                if s is not None:
                    by[j["role_label"]].append(s)
            f_mean, f_lo, f_hi = mean_ci(by["full"], rng)
            n_mean = st.mean(by["none"]) if by["none"] else None
            d_mean = st.mean(dfl) if dfl else None
            rows.append({"model": m, "lang": lang, "n_far": len(far), **{f"far_{k}": round(v, 3) for k, v in share.items()},
                         "score_default": d_mean, "score_far_full": f_mean, "ci_low": f_lo, "ci_high": f_hi,
                         "score_far_none": n_mean, "n_far_full": len(by["full"])})
            fmt = lambda x: "" if x is None else f"{x:.2f}"
            L.append(f"| {m.split('/')[-1]} | {lang} | {share['full']:.0%} | {share['somewhat']:.0%} | {share['none']:.0%} | "
                     f"{fmt(d_mean)} | {fmt(f_mean)} [{fmt(f_lo)}, {fmt(f_hi)}] (n={len(by['full'])}) | {fmt(n_mean)} | {len(far)} |")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "role_adoption.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "role_adoption.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    sheet = []
    for lang in ("ha", "yo"):
        pool = [j for j in J if j["lang"] == lang and j["group"] == "far"]
        rng.shuffle(pool)
        per = defaultdict(list)
        for j in pool:
            per[j["role_label"]].append(j)
        pick = []
        while len(pick) < a.sheet_size and any(per.values()):
            for k in ("full", "somewhat", "none"):
                if per[k] and len(pick) < a.sheet_size:
                    pick.append(per[k].pop())
        for j in pick:
            msgs = reply[(j["model"], j["conv_id"], lang)]
            sheet.append({"id": f"{j['model'].split('/')[-1]}|{j['conv_id']}|{lang}", "language": lang, "role": j["role"],
                          "user_message": msgs[0]["content"], "model_reply": msgs[1]["content"],
                          "your_rating (full/somewhat/none)": "", "understood_user (yes/partly/no)": "",
                          "notes": ""})
    sdir = a.out / "ratings"
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / "role_spotcheck_sheet.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0])); w.writeheader(); w.writerows(sheet)
    print(f"spot-check sheet: {len(sheet)} rows -> {sdir / 'role_spotcheck_sheet.csv'} (judge labels hidden)")


if __name__ == "__main__":
    main()
