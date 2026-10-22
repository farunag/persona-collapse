"""Pack replay + projection + langid JSONL files into one compact JSON for the workbench.

  python scripts/build_space_data.py --dir space/sample_data --model qwen --out space/data.json
  python scripts/build_space_data.py --dir <runs>/qwen/replay --model qwen --split-dir space/data
      (one file per conversation, space/data/<model>/<conv_id>.json, plus space/data/index.json)

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
    ap.add_argument("--out", type=Path)
    ap.add_argument("--split-dir", type=Path, help="write one file per conversation instead of --out")
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
            if r.get("stopped_at_turn"):
                c.setdefault("stopped", {}).setdefault(run, {})[r["lang"]] = r["stopped_at_turn"]
    out = {"model": meta.get("model"), "max_new_tokens": meta.get("max_new_tokens"), "label": a.label,
           "conversations": sorted(convs.values(), key=lambda c: (c["domain"], c["persona"], c["topic"]))}
    if a.split_dir:
        write_split(out, a.model, a.split_dir)
        return
    if not a.out:
        ap.error("give --out or --split-dir")
    dump(out, a.out)


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n = len(obj.get("conversations", []))
    print(f"{n} conversations -> {path} ({path.stat().st_size / 1e6:.2f} MB)")


def write_split(out, model_key, split_dir):
    """One file per conversation, space/data/<model>/<conv_id>.json, loaded on demand by the page,
    plus space/data/index.json listing every model, topic and conversation."""
    index_path = split_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"models": []}
    index["models"] = [m for m in index["models"] if m["key"] != model_key]
    domains = defaultdict(list)
    total = 0
    for c in out["conversations"]:
        path = split_dir / model_key / f"{c['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        total += path.stat().st_size
        domains[c["domain"]].append({"id": c["id"], "persona": c["persona"], "topic": c["topic"],
                                     "runs": {run: sorted(langs) for run, langs in c["runs"].items()}})
    index["models"].append({"key": model_key, "model": out["model"], "label": out["label"],
                            "max_new_tokens": out["max_new_tokens"], "domains": dict(domains)})
    index["models"].sort(key=lambda m: m["key"] != "qwen")
    dump(index, index_path)
    print(f"{len(out['conversations'])} conversation files in {split_dir / model_key} ({total / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
