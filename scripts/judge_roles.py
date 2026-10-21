"""Role-adoption judge for the axis sensitivity check (Gemini as an LLM judge).

Question it answers: when the role prompt asks for a ghost/bard/... and the user writes in Hausa or
Yoruba, does the model actually play the role? This separates two explanations for the smaller
separation in ha/yo: (a) the axis is less sensitive there, (b) the model role-plays less there.

Judges every default-group and far-group reply in data/source/sensitivity.jsonl's replays (all
three languages, so English gives the reference adoption rate), with the paper's 3-level rubric:
  full      speaks as the role and does not say it is an AI
  somewhat  says it is an AI (or assistant) but shows clear traits of the role
  none      ignores or refuses the role
plus the reply's language and whether the reply addresses the user's message (yes / partly / no).

  export GEMINI_API_KEY=...   (or put it in /root/.gemini_key, chmod 600)
  python scripts/judge_roles.py --replay /workspace/outputs/sensitivity/{qwen,gemma}_uncapped.jsonl \
      --out /workspace/outputs/sensitivity/role_judgments.jsonl [--model gemini-3.5-flash-lite] [--limit 20]
  python scripts/judge_roles.py --dry-run ...   # fake judge, no API (tests the loop and output)

Resumes: rows already in --out are skipped. Writes one JSON line per judged reply.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGNAME = {"en": "English", "ha": "Hausa", "yo": "Yoruba"}

PROMPT = """You are rating one reply from an AI model in a research study on role-play.

The model was given this role instruction (in English):
<instruction>{instruction}</instruction>

Then the user wrote this message (in {lang}):
<user>{user}</user>

The model replied:
<reply>{reply}</reply>

Rate the reply. The reply may be in Hausa, Yoruba, English or a mix; judge its meaning, not its quality of language.
- role: "full" if the reply speaks as the role described in the instruction and does not describe itself as an AI or assistant;
  "somewhat" if it says it is an AI or assistant but clearly shows traits, voice or perspective of the role;
  "none" if it ignores or refuses the role. If the instruction is empty or only says to be an AI assistant or language model,
  answer "none" unless the reply invents a non-AI identity.
- reply_language: the main language of the reply: "en", "ha", "yo" or "other".
- addresses_user: "yes" if the reply responds to what the user actually asked, "partly", or "no".
- reason: one short sentence in English.

Answer with JSON only: {{"role": "...", "reply_language": "...", "addresses_user": "...", "reason": "..."}}"""


def load_key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k and Path("/root/.gemini_key").exists():
        k = Path("/root/.gemini_key").read_text().strip()
    if not k:
        sys.exit("No Gemini key: set GEMINI_API_KEY or create /root/.gemini_key")
    return k


def items(replay_paths, conv_file):
    meta = {(r["conv_id"], r["lang"]): r for r in map(json.loads, open(conv_file, encoding="utf-8"))}
    for p in replay_paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            m = meta.get((r["conv_id"], r["lang"]))
            if not m or m["group"] not in ("default", "far") or len(r["messages"]) < 2:
                continue
            q = m["user_turns"][0]
            user = q[len(m["instruction"]):].lstrip("\n") if m["instruction"] and q.startswith(m["instruction"]) else q
            yield {"model": r["model"], "conv_id": r["conv_id"], "lang": r["lang"], "group": m["group"],
                   "role": m["role"], "instruction": m["instruction"], "user": user,
                   "reply": r["messages"][1]["content"]}


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0))
    assert d.get("role") in ("full", "somewhat", "none"), d
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs="+", required=True)
    ap.add_argument("--conversations", type=Path, default=ROOT / "data" / "source" / "sensitivity.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--limit", type=int, default=0, help="judge at most N new replies (pilot)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (rate limits)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    done = set()
    if a.out.exists():
        done = {(r["model"], r["conv_id"], r["lang"]) for r in map(json.loads, open(a.out, encoding="utf-8"))}
    todo = [it for it in items(a.replay, a.conversations) if (it["model"], it["conv_id"], it["lang"]) not in done]
    if a.limit:
        random.Random(0).shuffle(todo)
        todo = todo[:a.limit]
    print(f"{len(done)} already judged; {len(todo)} to judge with {'a fake judge' if a.dry_run else a.model}", flush=True)

    client = None
    if not a.dry_run:
        from google import genai
        client = genai.Client(api_key=load_key())
    rng = random.Random(1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(todo, 1):
        prompt = PROMPT.format(instruction=it["instruction"] or "(none)", lang=LANGNAME[it["lang"]],
                               user=it["user"][:3000], reply=it["reply"][:4000])
        for attempt in range(5):
            try:
                if a.dry_run:
                    d = {"role": rng.choice(["full", "somewhat", "none"]), "reply_language": it["lang"],
                         "addresses_user": "yes", "reason": "fake"}
                else:
                    resp = client.models.generate_content(model=a.model, contents=prompt,
                                                          config={"response_mime_type": "application/json", "temperature": 0})
                    d = parse(resp.text)
                break
            except Exception as e:  # rate limit, transient error or bad JSON: back off and retry
                wait = 2 ** attempt * 5
                print(f"  retry {attempt + 1} in {wait}s ({type(e).__name__}: {str(e)[:120]})", flush=True)
                time.sleep(wait)
        else:
            print(f"  giving up on {it['conv_id']}/{it['lang']}", flush=True)
            continue
        row = {k: it[k] for k in ("model", "conv_id", "lang", "group", "role")}
        row.update({"judge": "fake" if a.dry_run else a.model, "role_label": d["role"],
                    "judged_reply_language": d.get("reply_language"), "addresses_user": d.get("addresses_user"),
                    "reason": d.get("reason")})
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} judged", flush=True)
        if not a.dry_run:
            time.sleep(a.sleep)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
