"""Stage 2 settings shared by replay, project, langid and analyze.

Standard library only, so the laptop-side scripts (dry-run replay, langid tests, analyze)
import it without torch or numpy.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = ROOT / "data" / "source" / "conversations.jsonl"
RESULTS = ROOT / "results"
DEFAULT_OUT = Path("/workspace/outputs/replay")

LANGS = ("en", "ha", "yo")
DOMAINS = ("coding", "therapy", "metacognitive")

MODELS = {
    "qwen": {
        "name": "Qwen/Qwen3-32B",
        "axis_folder": "qwen-3-32b",
        "target_layer": 32,
        "n_layers": 64,
        "chat_kwargs": {"enable_thinking": False},
        "max_context": 32768,
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "gemma": {
        "name": "google/gemma-2-27b-it",
        "axis_folder": "gemma-2-27b",
        "target_layer": 22,
        "n_layers": 46,
        "chat_kwargs": {},
        "max_context": 8192,
        "capping_config": None,  # no published capping setting
        "capping_experiment": None,
    },
}


def replay_stem(model, capped):
    return f"{model}_{'capped' if capped else 'uncapped'}"


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
