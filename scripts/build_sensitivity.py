"""Build the axis-sensitivity check: does the Assistant Axis still separate role-play from the
default assistant when the user writes in Hausa or Yoruba?

Single-turn prompts = (role instruction, in English, from the paper's role set) + a user message.
The user messages are the FIRST user turns of 12 of our own conversations (4 per domain, spread
across personas), so Hausa and Yoruba use the native-checked translations. Every language gets
exactly the same role x message grid. The instruction is put at the start of the user turn
(Gemma 2's chat template has no system role), the same for both models.

Groups: default (3 neutral prompts from the paper: none, "You are an AI assistant.",
"You are a large language model."), near (analyst, consultant, evaluator: Assistant-like roles),
far (10 roles at the role-play end of the axis in the paper).

  python scripts/build_sensitivity.py --roles /workspace/assistant-axis/data/roles/instructions
  -> data/source/sensitivity.jsonl  (replay with: replay.py --conversations data/source/sensitivity.jsonl
                                      --domains sensitivity --max-new-tokens 300 --out /workspace/outputs/sensitivity)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAR = ["ghost", "hermit", "bard", "leviathan", "wraith", "trickster", "bohemian", "prophet", "oracle", "wanderer"]
NEAR = ["analyst", "consultant", "evaluator"]


def spread(items, k):
    n = len(items)
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", type=Path, required=True, help="assistant-axis data/roles/instructions")
    ap.add_argument("--per-domain", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "source" / "sensitivity.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "data" / "source" / "conversations.jsonl", encoding="utf-8")]
    by = {(r["conv_id"], r["lang"]): r for r in rows}
    picks = []
    for d in ("coding", "therapy", "metacognitive"):
        ids = sorted({(r["persona_id"], r["topic_id"], r["conv_id"]) for r in rows if r["domain"] == d})
        picks += [c for _, _, c in spread(ids, a.per_domain)]

    def first(role):
        return json.loads((a.roles / f"{role}.json").read_text(encoding="utf-8"))["instruction"]

    conds = [("default", f"default{i}", ins["pos"]) for i, ins in enumerate(first("default")[:3])]
    conds += [("near", r, first(r)[0]["pos"]) for r in NEAR] + [("far", r, first(r)[0]["pos"]) for r in FAR]

    out = []
    for ri, (group, role, ins) in enumerate(conds):
        for qi, cid in enumerate(picks):
            for lang in ("en", "ha", "yo"):
                q = by[(cid, lang)]["user_turns"][0]
                out.append({"conv_id": f"sens_{group}-{role}_q{qi:02d}", "domain": "sensitivity", "lang": lang,
                            "persona_id": ri, "topic_id": qi, "user_turns": [f"{ins}\n\n{q}" if ins else q],
                            "group": group, "role": role, "instruction": ins, "question_from": cid,
                            "source_file": "data/source/sensitivity.jsonl"})
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} prompts ({len(conds)} conditions x {len(picks)} messages x 3 languages) -> {a.out}")


if __name__ == "__main__":
    main()
