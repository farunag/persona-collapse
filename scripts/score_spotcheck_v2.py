"""Score both rounds of the native-speaker spot-check against the LLM judge and the axis score.

  python scripts/score_spotcheck_v2.py --sens <path to data/sensitivity.jsonl from the HF dataset>

Round 1 (far-role replies): results/ratings/human/{ha,yo}_annotator_*.csv
Round 2 (default replies):  results/ratings/human/default_{ha,yo}_annotator_*.csv
Writes results/spotcheck_round{1,2}_scored.csv (one row per reply, every annotator's labels, judge label,
score) and results/spotcheck_summary.md. ha_annotator_B's role labels are excluded (all 20 rated "full",
contradicting their own notes); their other labels are kept. Fluency is 1-5 and is left empty by annotators
when the reply is not in the user's language. Standard library only.
"""
import argparse, collections, csv, json, statistics
from pathlib import Path

ROUNDS = {1: ["ha_annotator_A", "ha_annotator_B", "ha_annotator_C", "ha_annotator_D",
              "yo_annotator_A", "yo_annotator_B"],
          2: ["default_ha_annotator_A", "default_ha_annotator_B", "default_yo_annotator_A", "default_yo_annotator_B"]}
ROLE_EXCLUDED = {"ha_annotator_B"}
MODELS = {"Qwen3-32B": "Qwen 3 32B", "gemma-2-27b-it": "Gemma 2 27B"}


def load(human, names):
    H = collections.defaultdict(dict)
    for n in names:
        for r in csv.DictReader(open(human / f"{n}.csv", encoding="utf-8")):
            H[tuple(r["full_id"].split("|"))][n] = r
    return H


def pct(a, b):
    return f"{a}/{b}" + (f" ({100 * a / b:.0f}%)" if b else "")


def pairs(H, names, field):
    """Exact agreement over every pair of annotators who rated the same reply."""
    same = tot = 0
    for v in H.values():
        got = [n for n in names if n in v and not (field == "role_played" and n in ROLE_EXCLUDED)]
        for i in range(len(got)):
            for j in range(i + 1, len(got)):
                tot += 1
                same += v[got[i]][field] == v[got[j]][field]
    return same, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=Path, required=True)
    ap.add_argument("--human", type=Path, default=Path("results/ratings/human"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    S = [json.loads(l) for l in open(a.sens, encoding="utf-8")]
    J = {(r["model"].split("/")[-1], r["conv_id"], r["language"]): r for r in S}

    # reference scores: mean score of every default and far-role reply per model and language
    ref = collections.defaultdict(list)
    for r in S:
        if r.get("score") not in (None, ""):
            ref[(r["model"].split("/")[-1], r["language"], r["group"])].append(float(r["score"]))
    md = ["# Native-speaker spot-check: both rounds", "",
          "Round 1: 40 far-role replies (asked to play a character). Round 2: 40 default replies (normal assistant).",
          "20 per language in each round, Qwen 3 32B and Gemma 2 27B, judge labels hidden from annotators.",
          "Fluency is 1 (not understandable) to 5 (native-like); annotators leave it empty when the reply is not",
          "in the user's language. Raw ratings: `ratings/human/`. Script: `scripts/score_spotcheck_v2.py`.", ""]
    for rnd, names in ROUNDS.items():
        H = load(a.human, names)
        rows = []
        for k, v in sorted(H.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
            j = J[k]
            row = {"model": k[0], "conv_id": k[1], "lang": k[2], "group": j["group"], "role": j["role"],
                   "score": round(float(j["score"]), 3), "judge_role": j["role_label"],
                   "judge_reply_language": j["judged_reply_language"], "judge_addresses_user": j["addresses_user"]}
            for n in names:
                if n in v:
                    s = n.replace("default_", "").replace("_annotator", "")
                    for f in ("role_played", "understood_user", "reply_language", "fluency"):
                        row[f"{s}_{f}"] = v[n][f]
            rows.append(row)
        out = a.out / f"spotcheck_round{rnd}_scored.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(f for r in rows for f in r)))
            w.writeheader(); w.writerows(rows)
        print(f"round {rnd}: {len(rows)} replies -> {out}")

        md += [f"## Round {rnd}: {'far-role' if rnd == 1 else 'default'} replies", "",
               "| Model | Language | Replies | Rated in the user's language | Fluency ≤ 2 (in-language ratings) "
               "| Mean fluency | Understood: yes / partly / no | Judge: addresses user yes | Mean axis score |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            for lang in ("ha", "yo"):
                ks = [k for k in H if k[0] == m and k[2] == lang]
                if not ks:
                    continue
                rs = [x for k in ks for x in H[k].values()]
                inl = sum(x["reply_language"] == lang for x in rs)
                fl = [int(x["fluency"]) for x in rs if x["fluency"] and x["reply_language"] == lang]
                und = collections.Counter(x["understood_user"] for x in rs)
                jy = sum(J[k]["addresses_user"] == "yes" for k in ks)
                sc = statistics.mean(float(J[k]["score"]) for k in ks)
                md.append(f"| {MODELS[m]} | {lang} | {len(ks)} | {pct(inl, len(rs))} | {pct(sum(f <= 2 for f in fl), len(fl))} "
                          f"| {statistics.mean(fl):.1f} | {und['yes']} / {und['partly']} / {und['no']} | {pct(jy, len(ks))} | {sc:+.2f} |"
                          if fl else
                          f"| {MODELS[m]} | {lang} | {len(ks)} | {pct(inl, len(rs))} | – | – | {und['yes']} / {und['partly']} / {und['no']} "
                          f"| {pct(jy, len(ks))} | {sc:+.2f} |")
        md += ["", "Counts in the understood column are ratings (each reply is rated by every annotator).", ""]
        md += ["Agreement between annotators (exact, over every pair rating the same reply):", ""]
        for lang in ("ha", "yo"):
            sub = {k: v for k, v in H.items() if k[2] == lang}
            ns = [n for n in names if f"{lang}_" in n]
            md.append(f"- {lang}: role {pct(*pairs(sub, ns, 'role_played'))}; understood {pct(*pairs(sub, ns, 'understood_user'))}; "
                      f"reply language {pct(*pairs(sub, ns, 'reply_language'))}; fluency {pct(*pairs(sub, ns, 'fluency'))}")
        if rnd == 1:
            same = tot = 0
            for k, v in H.items():
                for n, x in v.items():
                    if n not in ROLE_EXCLUDED:
                        tot += 1; same += J[k]["role_label"] == x["role_played"]
            md += ["", f"Judge role label vs annotator role label: {pct(same, tot)}."]
        # judge's 'addresses_user' against annotators who said 'no'
        over = [(k, n) for k, v in H.items() for n, x in v.items() if x["understood_user"] == "no" and J[k]["addresses_user"] == "yes"]
        nos = sum(x["understood_user"] == "no" for v in H.values() for x in v.values())
        md += ["", f"Judge said the reply addresses the user in {pct(len(over), nos)} of the ratings where an annotator said it did not.", ""]

    md += ["## Reference: mean axis score of all sensitivity replies", "",
           "| Model | Language | Default | Far-role | Gap |", "|---|---|---|---|---|"]
    for m in MODELS:
        for lang in ("en", "ha", "yo"):
            d, f = ref.get((m, lang, "default")), ref.get((m, lang, "far"))
            if d and f:
                md.append(f"| {MODELS[m]} | {lang} | {statistics.mean(d):+.2f} | {statistics.mean(f):+.2f} | "
                          f"{statistics.mean(d) - statistics.mean(f):.2f} |")
    (a.out / "spotcheck_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
