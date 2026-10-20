"""Build data/source/conversations.jsonl: the fixed user turns for every (conversation, language).

Reads the English originals (data/prior/raw/scaled-n60/<domain>/) and the Hausa / Yoruba
translations (data/prior/<domain>_translated/{ha,yo}/), keeps ONLY the user turns (the
assistant turns are old English Gemma replies and are never reused), and fills the 16
untranslated turns from data/prior/untranslated_turns.csv (user_turn there is 1-based).

Pure standard library, runs on the laptop:
  python scripts/build_conversations.py
"""

import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "data" / "prior"
OUT = ROOT / "data" / "source" / "conversations.jsonl"
DOMAINS = ("coding", "therapy", "metacognitive")
LANGS = ("en", "ha", "yo")
NAME_RE = re.compile(r"^(\w+?)_p(\d+)_t(\d+)_(\d+_\d+)(?:_(ha|yo))?\.json$")


def load_dir(domain, lang):
    d = PRIOR / "raw" / "scaled-n60" / domain if lang == "en" else PRIOR / f"{domain}_translated" / lang
    out = {}
    for f in sorted(d.glob("*.json")):
        m = NAME_RE.match(f.name)
        assert m and m.group(1) == domain, f"unexpected file name {f}"
        persona_id, topic_id = int(m.group(2)), int(m.group(3))
        data = json.loads(f.read_text(encoding="utf-8"))
        user_turns = [msg["content"] for msg in data["conversation"] if msg["role"] == "user"]
        conv_id = f"{domain}_p{persona_id}_t{topic_id}"
        assert conv_id not in out, f"duplicate {conv_id} in {d}"
        out[conv_id] = {"conv_id": conv_id, "domain": domain, "persona_id": persona_id, "topic_id": topic_id,
                        "lang": lang, "user_turns": user_turns,
                        "source_file": f.relative_to(ROOT).as_posix()}
    return out


def apply_fixes(rows):
    """Replace the untranslated turns listed in untranslated_turns.csv; returns the number applied."""
    by_file = {(r["lang"], Path(r["source_file"]).name): r for r in rows.values()}
    n = 0
    with open(PRIOR / "untranslated_turns.csv", encoding="utf-8-sig", newline="") as fh:
        for fix in csv.DictReader(fh):
            row = by_file[(fix["lang"], fix["file"])]
            i = int(fix["user_turn"]) - 1  # 1-based in the CSV
            # quoted CSV fields keep Windows line endings; the JSON turns use \n
            english, translation = (fix[k].replace("\r\n", "\n").strip() for k in ("english_text", "translation"))
            assert row["user_turns"][i].strip() == english, f"CSV text mismatch at {fix['file']} turn {i + 1}"
            assert translation, f"empty translation for {fix['file']} turn {i + 1}"
            row["user_turns"][i] = translation
            n += 1
    return n


def main():
    rows = []
    for domain in DOMAINS:
        per_lang = {lang: load_dir(domain, lang) for lang in LANGS}
        ids = set(per_lang["en"])
        for lang in LANGS:
            assert set(per_lang[lang]) == ids, f"{domain}/{lang}: conv_ids differ from en"
        rows.append((domain, per_lang))
    translated = {(lang, cid): r for _, per_lang in rows for lang in ("ha", "yo") for cid, r in per_lang[lang].items()}
    n_fixed = apply_fixes(translated)
    print(f"Filled {n_fixed} untranslated turns from untranslated_turns.csv")

    out_rows = []
    for domain, per_lang in rows:
        for cid in sorted(per_lang["en"], key=lambda c: (per_lang["en"][c]["persona_id"], per_lang["en"][c]["topic_id"])):
            en = per_lang["en"][cid]["user_turns"]
            for lang in LANGS:
                r = per_lang[lang][cid]
                assert len(r["user_turns"]) == len(en), f"{cid}/{lang}: {len(r['user_turns'])} vs {len(en)} user turns"
                assert all(t.strip() for t in r["user_turns"]), f"{cid}/{lang}: empty user turn"
                if lang != "en":
                    same = [i + 1 for i, (a, b) in enumerate(zip(r["user_turns"], en)) if a.strip() == b.strip()]
                    assert not same, f"{cid}/{lang}: user turns {same} identical to English"
                out_rows.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {OUT.relative_to(ROOT).as_posix()}")
    print(f"{'domain':<14}" + "".join(f"{lang:>22}" for lang in LANGS))
    for domain in DOMAINS:
        cells = []
        for lang in LANGS:
            rs = [r for r in out_rows if r["domain"] == domain and r["lang"] == lang]
            turns = [t for r in rs for t in r["user_turns"]]
            cells.append(f"{len(rs)} conv/{len(turns)} t/{st.mean(len(t) for t in turns):.0f}c")
        print(f"{domain:<14}" + "".join(f"{c:>22}" for c in cells))
    print("(conversations / user turns / mean characters per user turn)")
    for lang in LANGS:
        turns = [t for r in out_rows if r["lang"] == lang for t in r["user_turns"]]
        print(f"  {lang}: mean {st.mean(len(t) for t in turns):.1f} characters per user turn over {len(turns)} turns")
    assert len(out_rows) == 540, f"expected 540 rows, got {len(out_rows)}"


if __name__ == "__main__":
    main()
