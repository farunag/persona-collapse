"""Score the native-speaker role spot-check against the LLM judge.

  python scripts/score_spotcheck.py --sens <path to data/sensitivity.jsonl from the HF dataset>

Reads results/ratings/human/*.csv. Writes results/role_spotcheck_scored.csv (one row per reply with
judge label, score and every annotator's labels) and prints the agreement tables used in
results/role_spotcheck_report.md. ha_annotator_B's role labels are excluded from agreement (all 20 rated
"full", contradicting the annotator's own notes); their language and fluency labels are kept.
"""
import argparse, csv, json, collections
from pathlib import Path

ANN = ["yo_annotator_A", "yo_annotator_B", "ha_annotator_A", "ha_annotator_B"]
ROLE_EXCLUDED = {"ha_annotator_B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=Path, required=True)
    ap.add_argument("--human", type=Path, default=Path("results/ratings/human"))
    ap.add_argument("--out", type=Path, default=Path("results/role_spotcheck_scored.csv"))
    a = ap.parse_args()
    J = {(r["model"].split("/")[-1], r["conv_id"], r["language"]): r for r in map(json.loads, open(a.sens, encoding="utf-8"))}
    H = collections.defaultdict(dict)
    for n in ANN:
        for r in csv.DictReader(open(a.human / f"{n}.csv", encoding="utf-8")):
            H[tuple(r["full_id"].split("|"))][n] = r
    rows = []
    for k, v in sorted(H.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        j = J[k]
        row = {"model": k[0], "conv_id": k[1], "lang": k[2], "score": round(j["score"], 3),
               "judge_role": j["role_label"], "judge_reply_language": j["judged_reply_language"],
               "judge_addresses_user": j["addresses_user"]}
        for n in ANN:
            if n in v:
                s = n.replace("_annotator", "")
                row[f"{s}_role"], row[f"{s}_understood"] = v[n]["role_played"], v[n]["understood_user"]
                row[f"{s}_lang"], row[f"{s}_fluency"] = v[n]["reply_language"], v[n]["fluency"]
        rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} replies -> {a.out}")

    def agree(x, y, field="role_played"):
        ks = [k for k in H if x in H[k] and y in H[k]]
        return sum(H[k][x][field] == H[k][y][field] for k in ks), len(ks)
    print("\nAnnotator vs annotator (role):", {"yo": agree("yo_annotator_A", "yo_annotator_B"),
                                                "ha": agree("ha_annotator_A", "ha_annotator_B")})
    print("Judge vs annotator (role):")
    for n in ANN:
        ks = [k for k in H if n in H[k]]
        print(f"  {n}: {sum(J[k]['role_label'] == H[k][n]['role_played'] for k in ks)}/{len(ks)}"
              + ("  (excluded)" if n in ROLE_EXCLUDED else ""))
    print("\nBy group (annotators with valid role labels):")
    groups = collections.defaultdict(lambda: collections.Counter())
    for k, v in H.items():
        inlang = all(x["reply_language"] == k[2] for x in v.values())
        g = (k[0], k[2], "in-language" if inlang else "other/mixed", "judge " + J[k]["role_label"])
        for n, x in v.items():
            if n not in ROLE_EXCLUDED:
                groups[g]["human " + x["role_played"]] += 1
                groups[g]["understood " + x["understood_user"]] += 1
                if x["fluency"]:
                    groups[g]["fluency<=2"] += int(x["fluency"]) <= 2
    for g in sorted(groups):
        print(" ", g, dict(groups[g]))


if __name__ == "__main__":
    main()
