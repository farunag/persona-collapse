"""replay.py --dry-run: turn loop, batching, context stop, resume and output format, no GPU."""

import json

import pytest

import replay


def conv(cid, domain, lang, n_turns, chars):
    return {"conv_id": cid, "domain": domain, "persona_id": int(cid.split("_p")[1].split("_")[0]), "topic_id": 0,
            "lang": lang, "user_turns": [f"{lang} turn {i + 1} " + "x" * chars for i in range(n_turns)],
            "source_file": f"{cid}.json"}


@pytest.fixture
def source(tmp_path):
    rows = []
    for lang in ("en", "ha"):
        rows += [conv("coding_p0_t0", "coding", lang, 3, 40),
                 conv("coding_p1_t0", "coding", lang, 4, 40),
                 # long user turns: hits the context limit at turn 3
                 conv("therapy_p0_t0", "therapy", lang, 5, 400 if lang == "ha" else 40)]
    p = tmp_path / "conversations.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def run(source, out, *extra):
    return replay.main(["--dry-run", "--model", "gemma", "--langs", "en,ha", "--domains", "coding,therapy",
                        "--conversations", str(source), "--out", str(out), "--max-new-tokens", "100",
                        "--max-context", "400", *extra])


def read(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_loop_batching_and_context_stop(source, tmp_path):
    out = run(source, tmp_path / "o")
    rows = {(r["conv_id"], r["lang"]): r for r in read(out)}
    assert len(rows) == 6
    r = rows[("coding_p1_t0", "en")]
    assert r["turns_completed"] == 4 and r["stopped_at_turn"] is None
    assert [m["role"] for m in r["messages"]] == ["user", "assistant"] * 4
    assert all(m["content"] == "stub reply" for m in r["messages"][1::2])
    assert r["messages"][0]["content"].startswith("en turn 1")
    assert r["reply_tokens"] == [2] * 4 and r["finish_reasons"] == ["stop"] * 4
    assert r["prompt_tokens"] == sorted(r["prompt_tokens"])  # context grows every turn
    assert r["system_prompt"] is None and r["seed"] == 0 and r["sampling"]["max_new_tokens"] == 100
    assert r["model"] == "google/gemma-2-27b-it" and r["capped"] is False
    # ha therapy: (prompt + 100 new tokens) > 400 at some turn -> stopped there, earlier turns kept
    stop = rows[("therapy_p0_t0", "ha")]
    assert stop["stopped_at_turn"] is not None and stop["turns_completed"] == stop["stopped_at_turn"] - 1
    assert stop["prompt_tokens"][-1] + 100 <= 400
    assert rows[("therapy_p0_t0", "en")]["stopped_at_turn"] is None
    # batching: one generate call per turn over every active conversation
    log = read(out.with_name("gemma_uncapped.log.jsonl"))
    assert log[0]["active"] == 6
    assert [l["turn"] for l in log] == list(range(1, len(log) + 1))
    assert [l["active"] for l in log] == [6, 6, 5, 3, 1]  # ha therapy stops at 3; coding_p0 has 3 turns, p1 4
    # seeds: per request, stable
    assert r["request_seeds"][0] == replay.request_seed(0, "coding_p1_t0", "en", 1)


def test_resume_and_guards(source, tmp_path):
    out_dir = tmp_path / "o"
    run(source, out_dir, "--limit", "1")
    assert len(read(out_dir / "gemma_uncapped.jsonl")) == 4  # 1 per domain x 2 langs
    with pytest.raises(SystemExit):  # exists without --resume
        run(source, out_dir)
    with pytest.raises(SystemExit):  # different sampling settings on resume
        run(source, out_dir, "--resume", "--temperature", "1.0")
    run(source, out_dir, "--resume")
    rows = read(out_dir / "gemma_uncapped.jsonl")
    assert len(rows) == 6 and len({(r["conv_id"], r["lang"]) for r in rows}) == 6
    run(source, out_dir, "--resume")  # nothing left to do
    assert len(read(out_dir / "gemma_uncapped.jsonl")) == 6


def test_capped_is_qwen_only(source, tmp_path):
    with pytest.raises(SystemExit):
        run(source, tmp_path / "o", "--capped")


def test_real_conversations_file_selection():
    rows = replay.select_conversations(replay.CONVERSATIONS, ["en", "ha", "yo"], ["coding", "therapy", "metacognitive"], 2)
    assert len(rows) == 2 * 3 * 3
    assert {r["conv_id"] for r in rows if r["domain"] == "coding"} == {"coding_p0_t0", "coding_p0_t1"}
