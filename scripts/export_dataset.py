"""Flatten the raw run files into the public, viewer-friendly tables of the HF dataset.

Input  (--raw): a local copy of Faruna01/persona-collapse-runs, laid out as on the Hub:
    <model>/replay/<model>_<uncapped|capped>{.jsonl,.proj.jsonl,.langid.jsonl}
    sensitivity/<model>_uncapped{.jsonl,.proj.jsonl}   sensitivity/role_judgments.jsonl
Output (--out): data/turns-<model>_<condition>.jsonl   one row per (conversation, language, turn)
                data/sensitivity.jsonl                  one row per single-turn sensitivity reply

  python scripts/export_dataset.py --raw /path/to/runs --out /path/to/runs

The raw files stay as they are; only data/ is written. Standard library only.
"""

import argparse
import json
from pathlib import Path

RUNS = [("qwen", "uncapped"), ("qwen", "capped"), ("gemma", "uncapped")]


def read(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def key(r, turn=None):
    return (r["conv_id"], r["lang"], r["turn"] if turn is None else turn)


def layers(p):
    lp = p.get("layer_projections") or {}
    ids = sorted(lp, key=int)
    return [int(i) for i in ids], [lp[i] for i in ids]


def turn_rows(replay, proj, lid):
    proj = {key(p): p for p in proj}
    lid = {key(r): r for r in lid}
    for r in replay:
        msgs = r["messages"]
        for i in range(0, len(msgs) - 1, 2):
            t = i // 2 + 1
            p, l = proj.get(key(r, t), {}), lid.get(key(r, t), {})
            layer_ids, layer_vals = layers(p)
            yield {
                "model": r["model"], "model_revision": r["model_revision"],
                "condition": "capped" if r["capped"] else "uncapped", "capped": r["capped"],
                # "none", not null: a column that is null in every row of one file makes the
                # datasets loader infer type null and then fail on the capped file's strings
                "capping_experiment": r.get("capping_experiment") or "none",
                "language": r["lang"], "domain": r["domain"], "conv_id": r["conv_id"],
                "persona_id": r["persona_id"], "topic_id": r["topic_id"],
                "seed": r["seed"], "request_seed": (r.get("request_seeds") or [None] * t)[t - 1],
                "turn": t, "user_text": msgs[i]["content"], "reply_text": msgs[i + 1]["content"],
                "prompt_tokens": r["prompt_tokens"][t - 1], "reply_tokens": r["reply_tokens"][t - 1],
                "finish_reason": r["finish_reasons"][t - 1],
                "projection": p.get("projection"), "score": p.get("score"),
                "target_layer": p.get("target_layer"),
                "layer_ids": layer_ids, "layer_projections": layer_vals,
                "reply_language": l.get("detected"), "reply_language_label": l.get("label"),
                "reply_language_prob": l.get("prob"), "reply_is_code": l.get("code_only"),
                "reply_matches_user_language": l.get("matches_user"),
                "temperature": r["sampling"]["temperature"], "top_p": r["sampling"]["top_p"],
                "max_new_tokens": r["sampling"]["max_new_tokens"], "backend": r.get("backend"),
            }


def sensitivity_rows(raw, prompts):
    meta = {(m["conv_id"], m["lang"]): m for m in read(prompts)}
    judged = {(j["model"], j["conv_id"], j["lang"]): j for j in read(raw / "sensitivity" / "role_judgments.jsonl")}
    for model in ("qwen", "gemma"):
        stem = raw / "sensitivity" / f"{model}_uncapped"
        proj = {(p["conv_id"], p["lang"]): p for p in read(Path(f"{stem}.proj.jsonl"))}
        for r in read(Path(f"{stem}.jsonl")):
            m = meta.get((r["conv_id"], r["lang"]), {})
            p = proj.get((r["conv_id"], r["lang"]), {})
            j = judged.get((r["model"], r["conv_id"], r["lang"]), {})
            yield {
                "model": r["model"], "model_revision": r["model_revision"], "language": r["lang"],
                "conv_id": r["conv_id"], "group": m.get("group"), "role": m.get("role"),
                "instruction": m.get("instruction"), "question_from": m.get("question_from"),
                "user_text": r["messages"][0]["content"], "reply_text": r["messages"][1]["content"],
                "reply_tokens": r["reply_tokens"][0], "finish_reason": r["finish_reasons"][0],
                "projection": p.get("projection"), "score": p.get("score"),
                "judge": j.get("judge"), "role_label": j.get("role_label"),
                "judged_reply_language": j.get("judged_reply_language"),
                "addresses_user": j.get("addresses_user"), "judge_reason": j.get("reason"),
            }


def null_columns(path):
    """Columns that are null in every row of a file (they break schema inference across files)."""
    seen, nonnull = set(), set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            seen.update(row)
            nonnull.update(k for k, v in row.items() if v is not None)
    return sorted(seen - nonnull)


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = missing = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            n += 1
            missing += row["score"] is None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {n} rows, {missing} without a score")
    bad = null_columns(path)
    if bad:
        print(f"  WARNING: always null in this file: {bad}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=Path("data/source/sensitivity.jsonl"))
    a = ap.parse_args()
    for model, cond in RUNS:
        stem = a.raw / model / "replay" / f"{model}_{cond}"
        replay = read(stem.with_suffix(".jsonl"))
        if not replay:
            print(f"skip {stem}: not found")
            continue
        write(turn_rows(replay, read(Path(f"{stem}.proj.jsonl")), read(Path(f"{stem}.langid.jsonl"))),
              a.out / "data" / f"turns-{model}_{cond}.jsonl")
    write(sensitivity_rows(a.raw, a.prompts), a.out / "data" / "sensitivity.jsonl")


if __name__ == "__main__":
    main()
