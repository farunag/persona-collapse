"""Gate A: recompute per-turn Assistant-Axis projections on the paper's transcripts.

For each transcript, runs one forward pass, takes the mean residual activation over
each assistant turn's response tokens at the target layer (assistant_axis SpanMapper,
same as notebooks/project_transcipt.ipynb), and projects onto the published axis.
If a transcript carries a stored "projections" field, compares against it.

Usage:
  uv run --project /workspace/assistant-axis python scripts/gate_a.py --model qwen --files <json> [...]
  uv run --project /workspace/assistant-axis python scripts/gate_a.py --model qwen   # all Qwen transcripts
"""

import argparse
import glob
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import assistant_axis
from assistant_axis import get_config, load_axis, project
from assistant_axis.internals import ConversationEncoder, ProbingModel, SpanMapper

AXIS_REPO = "lu-christina/assistant-axis-vectors"
TRANSCRIPTS = Path("/workspace/assistant-axis/transcripts")
RESULTS = Path(__file__).resolve().parent.parent / "results"

MODELS = {
    "qwen": {
        "name": "Qwen/Qwen3-32B",
        "axis_folder": "qwen-3-32b",
        "chat_kwargs": {"enable_thinking": False},
        "max_tokens": None,
        "default_files": sorted(glob.glob(str(TRANSCRIPTS / "case_studies/qwen-3-32b/*.json"))),
    },
    "gemma": {
        "name": "google/gemma-2-27b-it",
        "axis_folder": "gemma-2-27b",
        "chat_kwargs": {},
        "max_tokens": 8192,
        "default_files": sorted(glob.glob(str(TRANSCRIPTS / "persona_drift/*.json"))),
    },
}


def label_for(path):
    p = Path(path)
    stem = p.name.split(".json")[0]
    return f"{p.parent.name}/{stem}"


def compute_projections(pm, encoder, span_mapper, conversation, axis, layer, chat_kwargs):
    """Per-assistant-turn projections via the library's span mapping + projection."""
    _, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
    roles = [s["role"] for s in spans]
    expected = [m["role"] for m in conversation if m["role"] != "system"]
    if roles != expected:
        raise RuntimeError(f"span/turn mismatch: got {len(roles)} spans for {len(expected)} turns")
    mean_acts = span_mapper.mean_all_turn_activations(pm, encoder, conversation, layer=layer, **chat_kwargs)
    assert mean_acts.shape[0] == len(spans)
    out = []
    for span, act in zip(spans, mean_acts):
        if span["role"] == "assistant":
            out.append({"n_tokens": span["n_tokens"], "projection": project(act, axis, layer=layer)})
    return out


def compare(ours, stored):
    if not stored:
        return {"stored_available": False}
    n = min(len(ours), len(stored))
    a, b = np.array(ours[:n]), np.array(stored[:n])
    return {
        "stored_available": True,
        "n_compared": n,
        "length_match": len(ours) == len(stored),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if n > 1 else None,
    }


def plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in results:
        y = [t["projection"] for t in r["turns"]]
        ax.plot(range(1, len(y) + 1), y, marker="o", markersize=3, label=r["label"])
    ax.set_xlabel("Assistant turn")
    ax.set_ylabel(f"Projection on Assistant Axis (layer {results[0]['layer']})")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--files", nargs="*", help="transcript JSONs (default: all for this model)")
    ap.add_argument("--out", help="output stem under results/ (default gate_a_<model>)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    files = args.files or spec["default_files"]
    stem = args.out or f"gate_a_{args.model}"
    cfg = get_config(spec["name"])
    layer = cfg["target_layer"]

    t0 = time.time()
    axis_path = hf_hub_download(AXIS_REPO, f"{spec['axis_folder']}/assistant_axis.pt", repo_type="dataset")
    axis = load_axis(axis_path)
    pm = ProbingModel(spec["name"], dtype=torch.bfloat16)
    encoder = ConversationEncoder(pm.tokenizer, spec["name"])
    span_mapper = SpanMapper(pm.tokenizer)
    print(f"Loaded {spec['name']} + axis {tuple(axis.shape)} in {time.time() - t0:.1f}s; layer {layer}")

    results = []
    for f in files:
        data = json.load(open(f))
        conv = data["conversation"]
        label = label_for(f)
        if data.get("model") != spec["name"]:
            print(f"!! {label}: transcript model is {data.get('model')}, not {spec['name']}")
        n_tok = len(pm.tokenizer.apply_chat_template(conv, tokenize=True, **spec["chat_kwargs"]))
        if spec["max_tokens"] and n_tok > spec["max_tokens"]:
            print(f"!! {label}: {n_tok} tokens exceeds context {spec['max_tokens']}, skipping")
            continue
        t1 = time.time()
        turns = compute_projections(pm, encoder, span_mapper, conv, axis, layer, spec["chat_kwargs"])
        dt = time.time() - t1
        ours = [t["projection"] for t in turns]
        cmp = compare(ours, data.get("projections"))
        print(f"{label}: {n_tok} tokens, {len(turns)} assistant turns, {dt:.1f}s; "
              f"range [{min(ours):.2f}, {max(ours):.2f}]; first {ours[0]:.2f} last {ours[-1]:.2f}; compare={cmp}")
        results.append({
            "label": label,
            "file": str(f),
            "transcript_model": data.get("model"),
            "domain": data.get("domain"),
            "steering": "capped" if "steering_config" in data else "unsteered",
            "steering_config": data.get("steering_config"),
            "n_tokens": n_tok,
            "layer": layer,
            "turns": turns,
            "comparison": cmp,
            "seconds": round(dt, 2),
        })
        torch.cuda.empty_cache()

    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": spec["name"],
        "model_revision": getattr(pm.model.config, "_commit_hash", None),
        "dtype": "bfloat16",
        "target_layer": layer,
        "axis_file": f"{AXIS_REPO}/{spec['axis_folder']}/assistant_axis.pt",
        "axis_path": axis_path,
        "chat_kwargs": spec["chat_kwargs"],
        "seed": None,  # forward passes only, no sampling
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "assistant_axis_path": os.path.dirname(assistant_axis.__file__),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "total_seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / f"{stem}.json"
    json.dump({"meta": meta, "results": results}, open(out_json, "w"), indent=2)
    plot(results, RESULTS / f"{stem}.png", f"Gate A: {spec['name']} per-turn projection")
    print(f"Wrote {out_json} and {stem}.png; total {meta['total_seconds']}s")

    del pm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
