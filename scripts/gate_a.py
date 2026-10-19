"""Gate A: recompute per-turn Assistant-Axis projections on the paper's transcripts.

For each transcript, runs one forward pass, takes the mean residual activation over
each assistant turn's response tokens at the target layer (assistant_axis SpanMapper,
same as notebooks/project_transcipt.ipynb), and projects onto the published axis.
If a transcript carries a stored "projections" field, compares against it.

Usage:
  uv run --project /workspace/assistant-axis python scripts/gate_a.py --model qwen --files <json> [...]
  uv run --project /workspace/assistant-axis python scripts/gate_a.py --model qwen   # all Qwen transcripts
  uv run --project /workspace/assistant-axis python scripts/gate_a.py --model qwen --from-json results/gate_a_qwen.json --out <stem>
      # re-summarise and re-plot saved projections (no GPU), adding normalised scores
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
from common import axis_reference, normalised_score

AXIS_REPO = "lu-christina/assistant-axis-vectors"
TRANSCRIPTS = Path("/workspace/assistant-axis/transcripts")
RESULTS = Path(__file__).resolve().parent.parent / "results"

MODELS = {
    "qwen": {
        "name": "Qwen/Qwen3-32B",
        "axis_folder": "qwen-3-32b",
        "chat_kwargs": {"enable_thinking": False},
        "max_tokens": None,
        "default_files": sorted(glob.glob(str(TRANSCRIPTS / "case_studies/qwen-3-32b/*.json")))
        + sorted(glob.glob(str(TRANSCRIPTS / "persona_drift/*.json"))),
    },
    "gemma": {
        "name": "google/gemma-2-27b-it",
        "axis_folder": "gemma-2-27b",
        "chat_kwargs": {},
        "max_tokens": 8192,
        "default_files": sorted(glob.glob(str(TRANSCRIPTS / "persona_drift/*.json"))),
    },
}


DEFAULT_PROJ_NOTE = ("default_projection / role_mean_projection: published default_vector.pt and the mean over "
                     "role_vectors/*.pt, projected onto the axis at the target layer; "
                     "score = (projection - role_mean) / (default - role_mean), 1 = default Assistant, 0 = average role")
HIGH_DOMAINS, LOW_DOMAINS = ("coding", "writing"), ("therapy", "philosophy")


def truncate(conversation, tokenizer, max_tokens, chat_kwargs):
    """Drop trailing user/assistant pairs until the conversation fits; returns (conv, n_tokens)."""
    conv = list(conversation)
    while conv:
        n = len(tokenizer.apply_chat_template(conv, tokenize=True, **chat_kwargs))
        if n <= max_tokens:
            return conv, n
        conv = conv[:-2]
    raise RuntimeError("no prefix fits the context")


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


def plot(results, path, title, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in ("case_studies", "persona_drift") if any(r["group"] == g for r in results)]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.5 * len(groups), 5.5), squeeze=False, sharey=True)
    domain_colors = {"coding": "tab:blue", "writing": "tab:green", "therapy": "tab:red", "philosophy": "tab:purple"}
    scenario_colors = {"jailbreak": "tab:orange", "delusion": "tab:brown", "selfharm": "tab:gray"}
    for ax, g in zip(axes[0], groups):
        for r in (r for r in results if r["group"] == g):
            y = [t["projection"] for t in r["turns"]]
            name = r["label"].split("/")[-1]
            if g == "persona_drift":
                style = dict(color=domain_colors.get(r["domain"]),
                             linestyle="-" if r["native"] else "--")
                name += f" ({r['transcript_model'].split('/')[-1]} text)"
            else:
                style = dict(color=scenario_colors.get(name.split("_")[0]),
                             linestyle="-" if r["steering"] == "capped" else ":")
            ax.plot(range(1, len(y) + 1), y, marker="o", markersize=3, label=name, **style)
        ax.axhline(ref["default_projection"], color="black", linewidth=1, alpha=0.6,
                   label=f"default vector ({ref['default_projection']:.1f})")
        ax.axhline(ref["role_mean_projection"], color="black", linewidth=1, alpha=0.6, linestyle="--",
                   label=f"mean role vector ({ref['role_mean_projection']:.1f})")
        ax.set_xlabel("Assistant turn")
        ax.set_title(g.replace("_", " "))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"Projection on Assistant Axis (layer {results[0]['layer']})")
    span = ref["default_projection"] - ref["role_mean_projection"]
    axes[0][-1].secondary_yaxis(
        "right", functions=(lambda p: normalised_score(p, ref),
                            lambda z: np.asarray(z, dtype=float) * span + ref["role_mean_projection"]),
    ).set_ylabel("Normalised score (1 = default Assistant, 0 = average role)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def summarize(results, ref, model_name):
    default_proj = ref["default_projection"]
    mean = {r["label"]: float(np.mean([t["projection"] for t in r["turns"]])) for r in results}
    score = {k: float(normalised_score(v, ref)) for k, v in mean.items()}
    out = {"reference": ref, "default_projection": default_proj, "mean_projection": mean, "mean_score": score}
    pairs = {}
    for r in results:
        if r["group"] == "case_studies" and r["steering"] == "capped":
            scen = r["label"].split("/")[-1].split("_")[0]
            uns = next((u for u in results if u["group"] == "case_studies" and u["steering"] == "unsteered"
                        and u["label"].split("/")[-1].startswith(scen + "_")), None)
            if uns:
                dc, du = abs(mean[r["label"]] - default_proj), abs(mean[uns["label"]] - default_proj)
                pairs[scen] = {"capped_mean": mean[r["label"]], "unsteered_mean": mean[uns["label"]],
                               "capped_score": score[r["label"]], "unsteered_score": score[uns["label"]],
                               "capped_dist_to_default": dc, "unsteered_dist_to_default": du,
                               "capped_closer": dc < du}
    if pairs:
        out["capping_pairs"] = pairs
    dom = {r["domain"]: mean[r["label"]] for r in results if r["group"] == "persona_drift"}
    if all(d in dom for d in HIGH_DOMAINS + LOW_DOMAINS):
        out["domain_means"] = dom
        out["domain_scores"] = {d: float(normalised_score(v, ref)) for d, v in dom.items()}
        out["domain_ordering_pass"] = min(dom[d] for d in HIGH_DOMAINS) > max(dom[d] for d in LOW_DOMAINS)
        out["domain_ordering_note"] = (
            "persona_drift transcripts are projected as written; text from "
            + ", ".join(f"{r['domain']}={r['transcript_model']}" for r in results if r["group"] == "persona_drift")
            + f". Only transcripts whose model is {model_name} are this model's own replies.")
    native = {r["label"]: mean[r["label"]] for r in results if r["group"] == "persona_drift" and r["native"]}
    if native:
        out["native_transcripts_mean"] = native
        out["native_transcripts_score"] = {k: score[k] for k in native}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--files", nargs="*", help="transcript JSONs (default: all for this model)")
    ap.add_argument("--out", help="output stem under results/ (default gate_a_<model>)")
    ap.add_argument("--from-json", help="re-summarise/re-plot a saved results JSON instead of running the model")
    args = ap.parse_args()

    spec = MODELS[args.model]
    files = args.files or spec["default_files"]
    stem = args.out or f"gate_a_{args.model}"
    cfg = get_config(spec["name"])
    layer = cfg["target_layer"]
    title = (f"Gate A: {spec['name']} per-turn projection\n"
             f"(dashed = another model's text, projected as written)")

    if args.from_json:
        if args.out is None:
            ap.error("--from-json needs --out so the saved results are not overwritten")
        saved = json.load(open(args.from_json))
        ref = axis_reference(spec["axis_folder"], layer)
        results = saved["results"]
        for r in results:
            for t in r["turns"]:
                t["score"] = float(normalised_score(t["projection"], ref))
        summary = summarize(results, ref, spec["name"])
        meta = dict(saved["meta"], rescored_from=args.from_json, default_projection_note=DEFAULT_PROJ_NOTE)
        out_json = RESULTS / f"{stem}.json"
        json.dump({"meta": meta, "summary": summary, "results": results}, open(out_json, "w"), indent=2)
        plot(results, RESULTS / f"{stem}.png", title, ref)
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_json} and {stem}.png")
        return

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
        n_tok_full = len(pm.tokenizer.apply_chat_template(conv, tokenize=True, **spec["chat_kwargs"]))
        n_asst_full = sum(m["role"] == "assistant" for m in conv)
        n_tok = n_tok_full
        if spec["max_tokens"] and n_tok_full > spec["max_tokens"]:
            conv, n_tok = truncate(conv, pm.tokenizer, spec["max_tokens"], spec["chat_kwargs"])
            print(f"!! {label}: {n_tok_full} tokens > {spec['max_tokens']}; truncated to {n_tok} tokens, "
                  f"kept {sum(m['role'] == 'assistant' for m in conv)}/{n_asst_full} assistant turns")
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
            "group": Path(f).parent.name if Path(f).parent.name == "persona_drift" else "case_studies",
            "native": data.get("model") == spec["name"],
            "steering": "capped" if "steering_config" in data else "unsteered",
            "steering_config": data.get("steering_config"),
            "n_tokens": n_tok,
            "n_tokens_full": n_tok_full,
            "assistant_turns_kept": len(turns),
            "assistant_turns_full": n_asst_full,
            "truncated": n_tok != n_tok_full,
            "layer": layer,
            "turns": turns,
            "comparison": cmp,
            "seconds": round(dt, 2),
        })
        torch.cuda.empty_cache()

    ref = axis_reference(spec["axis_folder"], layer, axis=axis)
    for r in results:
        for t in r["turns"]:
            t["score"] = float(normalised_score(t["projection"], ref))
    summary = summarize(results, ref, spec["name"])
    print(json.dumps(summary, indent=2))

    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": spec["name"],
        "model_revision": getattr(pm.model.config, "_commit_hash", None),
        "dtype": "bfloat16",
        "target_layer": layer,
        "axis_file": f"{AXIS_REPO}/{spec['axis_folder']}/assistant_axis.pt",
        "axis_path": axis_path,
        "chat_kwargs": spec["chat_kwargs"],
        "default_projection_note": DEFAULT_PROJ_NOTE,
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
    json.dump({"meta": meta, "summary": summary, "results": results}, open(out_json, "w"), indent=2)
    plot(results, RESULTS / f"{stem}.png", title, ref)
    print(f"Wrote {out_json} and {stem}.png; total {meta['total_seconds']}s")

    del pm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
