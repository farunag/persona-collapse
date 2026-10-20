"""Pack replay + projection + langid JSONL files into one compact JSON for the workbench.

  python scripts/build_space_data.py --dir space/sample_data --model qwen --out space/data.json

Standard library only. One entry per conversation, with every language and run
(uncapped / capped) it has: user turn, reply, normalised score, raw projection,
detected reply language, reply tokens and whether the reply hit the length limit.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="Pilot run: 2 conversations per topic")
    a = ap.parse_args()

    convs = {}
    meta = {}
    for run in ("uncapped", "capped"):
        stem = a.dir / f"{a.model}_{run}"
        reps = read(stem.with_suffix(".jsonl"))
        proj = {(p["conv_id"], p["lang"], p["turn"]): p for p in read(Path(f"{stem}.proj.jsonl"))}
        lid = {(r["conv_id"], r["lang"], r["turn"]): r for r in read(Path(f"{stem}.langid.jsonl"))}
        for r in reps:
            meta.setdefault("model", r["model"])
            meta.setdefault("max_new_tokens", r["sampling"]["max_new_tokens"])
            c = convs.setdefault(r["conv_id"], {"id": r["conv_id"], "domain": r["domain"],
                                                "persona": r["persona_id"], "topic": r["topic_id"], "runs": {}})
            msgs = r["messages"]
            turns = []
            for i in range(0, len(msgs) - 1, 2):
                t = i // 2 + 1
                p = proj.get((r["conv_id"], r["lang"], t), {})
                l = lid.get((r["conv_id"], r["lang"], t), {})
                turns.append({
                    "u": msgs[i]["content"], "a": msgs[i + 1]["content"],
                    "s": round(p["score"], 3) if "score" in p else None,
                    "p": round(p["projection"], 2) if "projection" in p else None,
                    "lang": l.get("detected"), "code": bool(l.get("code_only")),
                    "tok": r["reply_tokens"][t - 1], "cut": r["finish_reasons"][t - 1] == "length",
                })
            c["runs"].setdefault(run, {})[r["lang"]] = turns
    out = {"model": meta.get("model"), "max_new_tokens": meta.get("max_new_tokens"), "label": a.label,
           "conversations": sorted(convs.values(), key=lambda c: (c["domain"], c["persona"], c["topic"]))}
    a.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(out['conversations'])} conversations -> {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
