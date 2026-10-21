"""Axis sensitivity per model and language: how far does the score fall when the model is asked
to play a far role, compared with the default assistant, in English vs Hausa vs Yoruba?

  python scripts/analyze_sensitivity.py --proj /workspace/outputs/sensitivity/{qwen,gemma}_uncapped.proj.jsonl \
      [--langid ...langid.jsonl] --out results

Separation = mean score (default prompts) - mean score (far roles), per language; 95% bootstrap CI
over the 12 user messages (paired: the same messages in every condition). If Hausa/Yoruba
separation is close to English, the axis reads persona in those languages too, and low drift there
means the model stays near its assistant mode. If it is much smaller, the axis is less sensitive in
that language and low drift is weaker evidence. Standard library only.
"""

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def fmt_share(r):
    v = r["share_in_user_language"]
    return "" if v is None else f"{v:.0%}"


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", nargs="+", required=True)
    ap.add_argument("--langid", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    rows = [r for p in a.proj for r in read(p) if r["turn"] == 1]
    lid = {(r["model"], r["conv_id"], r["lang"]): r for p in a.langid for r in read(p)}
    # cell[(model, lang, group, q)] -> list of scores (one per role in the group)
    cell = defaultdict(list)
    roles = defaultdict(list)
    for r in rows:
        group, rest = r["conv_id"][5:].split("-", 1)
        role, q = rest.rsplit("_", 1)
        cell[(r["model"], r["lang"], group, q)].append(r["score"])
        roles[(r["model"], r["lang"], role, group)].append(r["score"])
    rng = random.Random(0)
    out_rows = []
    for m in sorted({k[0] for k in cell}):
        for lang in ("en", "ha", "yo"):
            qs = sorted({k[3] for k in cell if k[0] == m and k[1] == lang})
            if not qs:
                continue
            per_q = {g: {q: st.mean(cell[(m, lang, g, q)]) for q in qs if cell[(m, lang, g, q)]} for g in ("default", "near", "far")}
            sep_q = [per_q["default"][q] - per_q["far"][q] for q in qs]
            boots = sorted(st.mean(rng.choice(sep_q) for _ in sep_q) for _ in range(a.n_boot))
            n_lang = [lid[(m, r["conv_id"], lang)]["matches_user"] for r in rows
                      if r["model"] == m and r["lang"] == lang and (m, r["conv_id"], lang) in lid]
            out_rows.append({"model": m, "lang": lang, "n_messages": len(qs),
                             "default": round(st.mean(per_q["default"].values()), 3),
                             "near": round(st.mean(per_q["near"].values()), 3),
                             "far": round(st.mean(per_q["far"].values()), 3),
                             "separation": round(st.mean(sep_q), 3),
                             "ci_low": round(boots[int(.025 * len(boots))], 3), "ci_high": round(boots[int(.975 * len(boots)) - 1], 3),
                             "share_in_user_language": round(sum(n_lang) / len(n_lang), 3) if n_lang else None})
    for r in out_rows:
        en = next((x for x in out_rows if x["model"] == r["model"] and x["lang"] == "en"), None)
        r["separation_vs_en"] = round(r["separation"] / en["separation"], 2) if en and en["separation"] else None
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "sensitivity_separation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    role_rows = [{"model": m, "lang": l, "group": g, "role": ro, "mean_score": round(st.mean(v), 3), "n": len(v)}
                 for (m, l, ro, g), v in sorted(roles.items())]
    with open(a.out / "sensitivity_roles.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(role_rows[0]))
        w.writeheader(); w.writerows(role_rows)
    L = ["# Axis sensitivity check", "",
         "Single-turn replies to 12 of our user messages under 3 default prompts, 3 Assistant-like roles and",
         "10 far roles (English role instructions; user message in the language shown). Score 1 = default",
         "Assistant vector, 0 = mean role vector. Separation = default minus far, 95% CI over messages.", "",
         "| Model | Language | Default | Near roles | Far roles | Separation (95% CI) | vs English | Replies in user's language |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        L.append(f"| {r['model'].split('/')[-1]} | {r['lang']} | {r['default']:.2f} | {r['near']:.2f} | {r['far']:.2f} | "
                 f"{r['separation']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['separation_vs_en']} | "
                 f"{fmt_share(r)} |")
    (a.out / "sensitivity_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
