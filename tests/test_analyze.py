"""analyze.py on a small fake projection / langid / replay set with known answers."""

import json
import statistics as st

import pytest

import analyze

MODEL = "Qwen/Qwen3-32B"
# true per-language level by domain; drift(therapy) = therapy - coding
LEVEL = {("en", "coding"): 0.5, ("en", "therapy"): 0.3,
         ("ha", "coding"): 0.4, ("ha", "therapy"): 0.1,
         ("yo", "coding"): 0.6, ("yo", "therapy"): 0.4}
N_CONVS, N_TURNS = 12, 5


def fake_proj(capped=False, shift=0.0):
    rows = []
    for (lang, domain), level in LEVEL.items():
        for i in range(N_CONVS):
            # Yoruba therapy: only 8 conversations reach turn 5, so turn 5 is not comparable for therapy
            n_turns = 4 if (lang, domain) == ("yo", "therapy") and i >= 8 else N_TURNS
            for t in range(1, n_turns + 1):
                noise = 0.01 if i % 2 else -0.01  # symmetric: means stay exact
                rows.append({"conv_id": f"{domain}_p{i}_t0", "lang": lang, "domain": domain, "model": MODEL,
                             "model_revision": "abc", "capped": capped, "seed": 0, "turn": t,
                             "projection": 0.0, "score": level + shift + noise})
    return rows


def fake_langid():
    rows = []
    for (lang, domain) in LEVEL:
        for i in range(N_CONVS):
            for t in range(1, N_TURNS + 1):
                # Hausa therapy: every 4th conversation answers in English
                detected = "en" if (lang, domain) == ("ha", "therapy") and i % 4 == 0 else lang
                rows.append({"conv_id": f"{domain}_p{i}_t0", "lang": lang, "domain": domain, "model": MODEL,
                             "capped": False, "turn": t, "detected": detected, "matches_user": detected == lang,
                             "code_only": False})
    return rows


def fake_replay():
    rows = []
    for (lang, domain) in LEVEL:
        for i in range(N_CONVS):
            rows.append({"conv_id": f"{domain}_p{i}_t0", "lang": lang, "domain": domain, "model": MODEL,
                         "capped": False, "reply_tokens": [100, 300, 400], "finish_reasons": ["stop", "stop", "length"],
                         "prompt_tokens": [50, 500, 900], "stopped_at_turn": 4 if (lang == "ha" and i == 0) else None})
    return rows


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path):
    return {"proj": [write(tmp_path / "q_uncapped.proj.jsonl", fake_proj()),
                     write(tmp_path / "q_capped.proj.jsonl", fake_proj(capped=True, shift=0.2))],
            "langid": [write(tmp_path / "q.langid.jsonl", fake_langid())],
            "replay": [write(tmp_path / "q.jsonl", fake_replay())],
            "out": tmp_path / "results"}


def run(files, **kw):
    kw.setdefault("plots", False)
    return analyze.analyze(files["proj"], files["langid"], files["replay"], files["out"], min_convs=10,
                           n_boot=300, seed=0, **kw)


def test_comparable_turns(files):
    res = run(files)
    comp = res["comparable"]
    assert comp[(("Qwen3-32B", False), "coding")] == [1, 2, 3, 4, 5]
    assert comp[(("Qwen3-32B", False), "therapy")] == [1, 2, 3, 4]  # yo therapy: 8 < 10 at turn 5


def test_baselines_and_drift(files):
    res = run(files)
    for lang in ("en", "ha", "yo"):
        assert st.mean(res["baselines"][(("Qwen3-32B", False), lang)]) == pytest.approx(LEVEL[(lang, "coding")])
    for r in res["turns"]:
        if not r["capped"]:
            assert r["drift"] == pytest.approx(LEVEL[(r["lang"], r["domain"])] - LEVEL[(r["lang"], "coding")])
            assert r["ci_lo"] <= r["mean_score"] <= r["ci_hi"]
            assert r["drift_lo"] <= r["drift"] <= r["drift_hi"]
    assert not any(r["domain"] == "therapy" and r["turn"] == 5 for r in res["turns"])
    dom = {(r["lang"], r["domain"], r["capped"]): r for r in res["domains"]}
    assert dom[("ha", "therapy", False)]["drift"] == pytest.approx(-0.3)
    # capping shifts every score by +0.2, so drift (relative to its own coding baseline) is unchanged
    assert dom[("ha", "therapy", True)]["mean_score"] == pytest.approx(0.3)
    assert dom[("ha", "therapy", True)]["drift"] == pytest.approx(-0.3)


def test_gap_vs_english(files):
    gaps = {(g["lang"], g["domain"], g["capped"]): g for g in run(files)["gaps"]}
    assert gaps[("ha", "therapy", False)]["gap_vs_en"] == pytest.approx(-0.1)
    assert gaps[("yo", "therapy", False)]["gap_vs_en"] == pytest.approx(0.0, abs=1e-9)
    assert gaps[("ha", "coding", False)]["gap_vs_en"] == pytest.approx(0.0, abs=1e-9)


def test_reply_language(files):
    res = run(files)
    over = {(r["lang"], r["domain"]): r for r in res["reply_language"]}
    assert over[("ha", "therapy")]["share_user_lang"] == pytest.approx(0.75)
    assert over[("ha", "therapy")]["detected_en"] == 3 * N_TURNS
    assert over[("en", "coding")]["share_user_lang"] == 1.0
    # by-turn table keeps comparable turns only
    assert {r["turn"] for r in res["reply_language_by_turn"] if r["domain"] == "therapy"} == {1, 2, 3, 4}
    assert any("only 75% of replies" in f for f in res["flags"])


def test_tokens_and_flags(files):
    res = run(files)
    tok = {(r["lang"], r["domain"]): r for r in res["tokens"]}
    assert tok[("en", "coding")]["mean_reply_tokens"] == pytest.approx(800 / 3)
    assert tok[("en", "coding")]["share_hit_max_new_tokens"] == pytest.approx(1 / 3)
    assert tok[("ha", "coding")]["context_stops"] == 1
    assert any("coding baselines differ" in f for f in res["flags"])  # 0.4 vs 0.6
    assert any("cut at max_new_tokens" in f for f in res["flags"])


def test_outputs_and_no_overwrite(files):
    run(files)
    out = files["out"]
    md = (out / "stage2_summary.md").read_text(encoding="utf-8")
    assert "Per-language coding baselines" in md and "| Qwen3-32B uncapped | therapy | ha |" in md
    assert (out / "stage2_turn_scores.csv").exists()
    with pytest.raises(SystemExit):
        run(files)
    run(files, overwrite=True)


def test_bootstrap_constant_and_small():
    import random
    rng = random.Random(0)
    assert analyze.bootstrap_ci(st.mean, [[0.5] * 5], 100, rng) == (0.5, 0.5)
    assert analyze.bootstrap_ci(st.mean, [[0.5]], 100, rng) == (None, None)


def test_plots(files):
    pytest.importorskip("matplotlib")
    res = run(files, plots=True)
    for name in ("stage2_drift_Qwen3-32B_uncapped.png", "stage2_capping_Qwen3-32B.png",
                 "stage2_reply_language_Qwen3-32B_uncapped.png"):
        assert (files["out"] / name).exists()
