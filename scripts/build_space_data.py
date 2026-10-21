"""Pack replay + projection + langid JSONL files into one compact JSON for the workbench.

  python scripts/build_space_data.py --dir space/sample_data --model qwen --out space/data.json
  python scripts/build_space_data.py --dir <runs>/qwen/replay --model qwen --split-dir space/data
      (one file per topic, space/data/<model>_<domain>.json, plus space/data/index.json)

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
    ap.add_argument("--split-dir", type=Path, help="write one file per topic instead of --out")
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
    """space/data/<model>_<domain>.json per topic, and index.json listing every file."""
    index_path = split_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"files": []}
    index["files"] = [f for f in index["files"] if f["model_key"] != model_key]
    by_domain = defaultdict(list)
    for c in out["conversations"]:
        by_domain[c["domain"]].append(c)
    for domain, convs in sorted(by_domain.items()):
        name = f"{model_key}_{domain}.json"
        dump({**{k: v for k, v in out.items() if k != "conversations"}, "domain": domain, "conversations": convs},
             split_dir / name)
        runs = sorted({r for c in convs for r in c["runs"]})
        index["files"].append({"model_key": model_key, "model": out["model"], "domain": domain, "file": name,
                               "conversations": len(convs), "runs": runs})
    index["files"].sort(key=lambda f: (f["model_key"], f["domain"]))
    dump(index, index_path)


if __name__ == "__main__":
    main()
