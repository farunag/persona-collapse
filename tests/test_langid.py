"""langid.py with a stub predictor in place of GlotLID (fastText cannot load on the laptop)."""

import json

import langid


def stub_predictor(text):
    """Keyword stub with GlotLID's label format."""
    low = text.lower()
    if "sannu" in low:
        return "__label__hau_Latn", 0.97
    if "báwo" in low:
        return "__label__yor_Latn", 0.95
    if "def " in low or "import" in low:
        return "__label__xxx_Zyyy", 0.40  # code-like text
    if "bonjour" in low:
        return "__label__fra_Latn", 0.90
    return "__label__eng_Latn", 0.99


def test_map_label():
    assert langid.map_label("__label__eng_Latn") == "en"
    assert langid.map_label("__label__hau_Latn") == "ha"
    assert langid.map_label("yor_Latn") == "yo"
    assert langid.map_label("__label__fra_Latn") == "other"


def test_strip_code():
    text = "Sannu! Ga misali:\n```python\nimport os\ndef f():\n    pass\n```\nKa yi amfani da `os.path`."
    prose = langid.strip_code(text)
    assert "import" not in prose and "os.path" not in prose and "Sannu" in prose
    assert langid.strip_code("```\nunclosed fence\nimport x") .strip() == ""


def test_detect_prose_code_and_empty():
    d = langid.detect("Sannu, ga lambar:\n```\nimport numpy\n```", stub_predictor)
    assert d["detected"] == "ha" and not d["code_only"] and d["prob"] == 0.97
    d = langid.detect("```python\nimport numpy\n```", stub_predictor)
    assert d["code_only"] and d["detected"] == "other"
    d = langid.detect("   ", stub_predictor)
    assert d["detected"] == "empty"


def test_run_writes_one_row_per_reply(tmp_path):
    rec = {"model": "google/gemma-2-27b-it", "capped": False, "seed": 0, "model_revision": "r", "domain": "therapy"}
    replay = [
        {**rec, "conv_id": "therapy_p0_t0", "lang": "ha",
         "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Sannu da zuwa"},
                      {"role": "user", "content": "u"}, {"role": "assistant", "content": "Hello, how can I help?"}]},
        {**rec, "conv_id": "therapy_p0_t0", "lang": "yo",
         "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Báwo ni"},
                      {"role": "user", "content": "u"}, {"role": "assistant", "content": "Bonjour"}]},
    ]
    src = tmp_path / "gemma_uncapped.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replay), encoding="utf-8")
    out = tmp_path / "gemma_uncapped.langid.jsonl"
    assert langid.run(src, out, stub_predictor, "test") == 4
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["lang"], r["turn"], r["detected"], r["matches_user"]) for r in rows] == [
        ("ha", 1, "ha", True), ("ha", 2, "en", False), ("yo", 1, "yo", True), ("yo", 2, "other", False)]
    assert rows[0]["detector"] == "cis-lmu/glotlid@test" and rows[0]["label"] == "hau_Latn"
