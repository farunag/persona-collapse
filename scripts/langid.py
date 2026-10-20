"""Detect the language of every model reply with GlotLID (cis-lmu/glotlid, fastText).

Stores the top GlotLID label and probability per reply, mapped to en / ha / yo / other, and
whether it matches the user's language. Code (``` blocks and `inline`) is removed before
detection, since code is not in any natural language; a reply that is only code keeps its full
text and is flagged code_only.

Output: <replay>.langid.jsonl next to the replay file, one line per reply.

Usage:
  uv run --project /workspace/assistant-axis python scripts/langid.py --replay /workspace/outputs/replay/qwen_uncapped.jsonl
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from stage2_config import append_jsonl, read_jsonl

GLOTLID_REPO = "cis-lmu/glotlid"
LABEL_MAP = {"eng_Latn": "en", "hau_Latn": "ha", "yor_Latn": "yo"}
FENCE_RE = re.compile(r"```.*?(```|$)", re.S)
INLINE_RE = re.compile(r"`[^`\n]+`")


def strip_code(text):
    return INLINE_RE.sub(" ", FENCE_RE.sub(" ", text))


def map_label(label):
    """'__label__hau_Latn' -> 'ha'; anything outside en/ha/yo -> 'other'."""
    return LABEL_MAP.get(label.replace("__label__", ""), "other")


class GlotLID:
    def __init__(self):
        import fasttext
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GLOTLID_REPO, "model.bin")
        self.revision = Path(path).parent.name
        self.model = fasttext.load_model(path)

    def __call__(self, text):
        """(label, probability) of the top prediction."""
        text = " ".join(text.split())  # fastText rejects newlines
        # the low-level predict avoids fasttext's numpy wrapper, which breaks on numpy 2
        (prob, label), = self.model.f.predict(text, 1, 0.0, "strict")
        return label, float(prob)


def detect(text, predictor):
    prose = strip_code(text)
    code_only = not prose.strip()
    if not text.strip():
        return {"label": None, "prob": None, "detected": "empty", "code_only": False, "chars_used": 0}
    label, prob = predictor(text if code_only else prose)
    return {"label": label.replace("__label__", ""), "prob": round(prob, 4), "detected": map_label(label),
            "code_only": code_only, "chars_used": len(text if code_only else prose)}


def run(replay_path, out_path, predictor, revision):
    recs = read_jsonl(replay_path)
    created = datetime.now(timezone.utc).isoformat()
    n = 0
    for rec in recs:
        replies = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
        for turn, text in enumerate(replies, start=1):
            d = detect(text, predictor)
            append_jsonl(out_path, {
                "conv_id": rec["conv_id"], "lang": rec["lang"], "domain": rec["domain"], "model": rec["model"],
                "capped": rec["capped"], "seed": rec["seed"], "model_revision": rec["model_revision"],
                "turn": turn, **d, "matches_user": d["detected"] == rec["lang"],
                "detector": f"{GLOTLID_REPO}@{revision}", "created": created,
            })
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: <replay>.langid.jsonl")
    args = ap.parse_args()
    out_path = args.out or args.replay.with_suffix(".langid.jsonl")
    if out_path.exists():
        ap.error(f"{out_path} exists; remove it or choose another --out")
    predictor = GlotLID()
    n = run(args.replay, out_path, predictor, predictor.revision)
    rows = read_jsonl(out_path)
    for lang in sorted({r["lang"] for r in rows}):
        rs = [r for r in rows if r["lang"] == lang]
        share = sum(r["matches_user"] for r in rs) / len(rs)
        print(f"{lang}: {len(rs)} replies, {share:.1%} in the user's language")
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
