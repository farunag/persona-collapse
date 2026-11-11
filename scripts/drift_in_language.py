"""Drift when the model actually replies in the user's language.

Recomputes the uncapped drift table (results/stage2_domain_drift.csv) keeping only turns whose reply is
in the user's language (or is code), so a switch to English cannot drive the result. Same comparable
turn windows, same per-language coding baseline, same bootstrap (2,000 resamples over conversations).
Also prints the per-turn drift for Gemma, to see when it starts. The "in-language prose" filter drops
replies containing code blocks everywhere, baseline included, so code in the coding baseline cannot drive
the result.

  python scripts/drift_in_language.py            # loads Faruna01/persona-collapse-runs ("turns")
  python scripts/drift_in_language.py --local data/   # or local turns-*.jsonl files

Writes results/drift_in_language.csv and results/drift_in_language.md.
"""
import argparse, collections, csv, json, random, re, statistics
from pathlib import Path

REPO = "Faruna01/persona-collapse-runs"


def load(local):
    if local:
        rows = []
        for f in sorted(Path(local).glob("turns-*_uncapped.jsonl")):
            rows += [json.loads(l) for l in open(f, encoding="utf-8")]
        return rows
    from datasets import load_dataset
    ds = load_dataset(REPO, "turns")
    return [r for split in ds.values() for r in split if r["condition"] == "uncapped"]


def ci(vals, rng, n=2000):
    bs = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None)
    ap.add_argument("--windows", default="results/stage2_domain_drift.csv")
    a = ap.parse_args()
    win = {}
    for r in csv.DictReader(open(a.windows, encoding="utf-8")):
        if r["capped"] == "False":
            lo, hi = map(int, r["turns"].split("-"))
            win[(r["model"], r["domain"])] = (lo, hi)  # same window for every language of a model/domain
    rows = load(a.local)
    ok = lambda r: bool(r["reply_matches_user_language"]) or bool(r["reply_is_code"])
    conv = collections.defaultdict(lambda: collections.defaultdict(list))  # (model,lang,domain,filter) -> conv -> scores
    turn = collections.defaultdict(list)
    kept = collections.Counter(); total = collections.Counter(); code = collections.Counter(); cb = collections.defaultdict(list)
    for r in rows:
        m = r["model"].split("/")[-1]
        w = win.get((m, r["domain"]))
        if not w or r["score"] is None or not (w[0] <= r["turn"] <= w[1]):
            continue
        total[(m, r["language"], r["domain"])] += 1
        conv[(m, r["language"], r["domain"], "all")][r["conv_id"]].append(r["score"])
        has_code = bool(re.search(r"```|^(    |\t)\S", r.get("reply_text") or "", re.M))
        if bool(r["reply_matches_user_language"]) and not has_code:
            conv[(m, r["language"], r["domain"], "in-language prose")][r["conv_id"]].append(r["score"])
        if r["domain"] == "coding" and bool(r["reply_matches_user_language"]):
            cb[(m, r["language"], has_code)].append(r["score"])
        if r["reply_is_code"]:
            code[(m, r["language"], r["domain"])] += 1
        if ok(r):
            kept[(m, r["language"], r["domain"])] += 1
            conv[(m, r["language"], r["domain"], "in-language")][r["conv_id"]].append(r["score"])
            turn[(m, r["language"], r["domain"], r["turn"])].append(r["score"])
    rng = random.Random(0)
    out = []
    for m in sorted({k[0] for k in conv}):
        for flt in ("all", "in-language", "in-language prose"):
            for lang in ("en", "ha", "yo"):
                bk = conv.get((m, lang, "coding", flt), {})
                bvals = [statistics.mean(v) for v in bk.values()]
                if not bvals:
                    continue
                base = statistics.mean(bvals)
                for dom in ("therapy", "metacognitive"):
                    ck = conv.get((m, lang, dom, flt), {})
                    vals = [statistics.mean(v) - base for v in ck.values()]
                    if len(vals) < 5:
                        continue
                    lo, hi = ci(vals, rng)
                    out.append({"model": m, "filter": flt, "lang": lang, "domain": dom, "n_convs": len(vals),
                                "share_turns_kept": round(kept[(m, lang, dom)] / max(total[(m, lang, dom)], 1), 3),
                                "baseline": round(base, 3), "drift": round(statistics.mean(vals), 3),
                                "drift_lo": round(lo, 3), "drift_hi": round(hi, 3)})
    md_code = ["", "## In-language coding turns with and without code blocks", "",
               "| Model | Language | With code: turns, mean score | Without code: turns, mean score |", "|---|---|---|---|"]
    for m in sorted({k[0] for k in cb}):
        for lang in ("en", "ha", "yo"):
            w, wo = cb.get((m, lang, True), []), cb.get((m, lang, False), [])
            f = lambda v: f"{len(v)}, {statistics.mean(v):+.2f}" if v else "0, –"
            md_code.append(f"| {m} | {lang} | {f(w)} | {f(wo)} |")
    Path("results").mkdir(exist_ok=True)
    with open("results/drift_in_language.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    md = ["# Drift with replies in the user's language only", "",
          "| Model | Turns used | Language | Domain | Conversations | Turns kept | Drift [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['model']} | {r['filter']} | {r['lang']} | {r['domain']} | {r['n_convs']} | "
                  f"{r['share_turns_kept'] if r['filter'] == 'in-language' else 1.0:.0%} | "
                  f"{r['drift']:+.2f} [{r['drift_lo']:+.2f}, {r['drift_hi']:+.2f}] |")
    md += ["", "## Gemma per-turn drift, in-language replies (mean score minus that language's coding baseline)", "",
           "| Language | Domain | " + " | ".join(f"t{t}" for t in range(1, 9)) + " |", "|---|---|" + "---|" * 8]
    for lang in ("en", "ha", "yo"):
        base = next((r["baseline"] for r in out if r["model"].startswith("gemma") and r["filter"] == "in-language"
                     and r["lang"] == lang), None)
        if base is None:
            continue
        for dom in ("therapy", "metacognitive"):
            cells = []
            for t in range(1, 9):
                v = turn.get(("gemma-2-27b-it", lang, dom, t), [])
                cells.append(f"{statistics.mean(v) - base:+.2f} ({len(v)})" if len(v) >= 5 else "–")
            md.append(f"| {lang} | {dom} | " + " | ".join(cells) + " |")
    md += md_code
    Path("results/drift_in_language.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
