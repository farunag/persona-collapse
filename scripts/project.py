"""Project every replayed reply onto the Assistant Axis (no capping hooks, also for capped transcripts).

For each finished conversation: one forward pass with the HF model (ProbingModel, bfloat16),
mean post-MLP residual activation over each reply's tokens (assistant_axis ConversationEncoder
spans + SpanMapper, the gate_a.py method, batched with ActivationExtractor.batch_conversations),
projection onto the published axis at the target layer, and the normalised score from
scripts/common.py. Also stores the projection at every 4th layer (workbench heatmap).

Output: <replay>.proj.jsonl next to the replay file, one line per (conversation, reply turn) with
model, lang, domain, seed, capped, turn, projection, score, layer_projections, n_tokens.

Usage:
  uv run --project /workspace/assistant-axis python scripts/project.py --replay /workspace/outputs/replay/qwen_uncapped.jsonl
  ... --check   # first conversation: batched vs single-conversation (gate_a) projections, then exit
"""

import argparse
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import assistant_axis
from assistant_axis import load_axis, project
from assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel, SpanMapper
from common import AXIS_REPO, axis_reference, normalised_score
from stage2_config import MODELS, append_jsonl, read_jsonl


def per_turn_activations(extractor, encoder, span_mapper, convs, layers, max_length, chat_kwargs):
    """Per-conversation (n_turns, n_layers, hidden) mean activations + spans, one batched forward pass."""
    acts, meta = extractor.batch_conversations(convs, layer=layers, max_length=max_length, **chat_kwargs)
    _, batch_spans, _ = encoder.build_batch_turn_spans(convs, **chat_kwargs)
    per_conv = span_mapper.map_spans(acts, batch_spans, meta)
    out = []
    for i, conv in enumerate(convs):
        spans = sorted((s for s in batch_spans if s["conversation_id"] == i), key=lambda s: s["turn"])
        roles = [s["role"] for s in spans]
        if roles != [m["role"] for m in conv]:
            raise RuntimeError(f"span/turn mismatch: {len(roles)} spans for {len(conv)} messages")
        if per_conv[i].shape[0] != len(spans):
            raise RuntimeError(f"{per_conv[i].shape[0]} turn activations for {len(spans)} spans (truncated?)")
        out.append((per_conv[i].float().cpu(), spans))
    del acts
    return out


def log_error(out_path, rec, err):
    """Skip a conversation that cannot be projected; record why next to the output."""
    print(f"!! {rec['conv_id']}/{rec['lang']}: not projected: {err}", flush=True)
    append_jsonl(out_path.with_suffix(".errors.jsonl"), {"conv_id": rec["conv_id"], "lang": rec["lang"],
                                                          "error": str(err)})


def rows_for(rec, acts, spans, axis, layers, target, ref, common):
    rows, turn = [], 0
    li = {L: j for j, L in enumerate(layers)}
    for act, span in zip(acts, spans):
        if span["role"] != "assistant":
            continue
        turn += 1
        proj = project(act[li[target]], axis, layer=target)
        rows.append({
            "conv_id": rec["conv_id"], "lang": rec["lang"], "domain": rec["domain"],
            "persona_id": rec["persona_id"], "topic_id": rec["topic_id"],
            "model": rec["model"], "model_revision": rec["model_revision"], "capped": rec["capped"],
            "seed": rec["seed"], "turn": turn, "projection": proj, "score": float(normalised_score(proj, ref)),
            "n_tokens": span["n_tokens"], "reply_tokens_generated": rec["reply_tokens"][turn - 1],
            "finish_reason": rec["finish_reasons"][turn - 1],
            "layer_projections": {str(L): project(act[li[L]], axis, layer=L) for L in layers},
            **common,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True, help="replay JSONL from replay.py")
    ap.add_argument("--out", type=Path, help="default: <replay>.proj.jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--layer-step", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N conversations only (tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare batched vs single-conversation projections and exit")
    args = ap.parse_args()

    recs = read_jsonl(args.replay)
    keys = {r["model_key"] for r in recs}
    assert len(keys) == 1, f"replay file mixes models {keys}"
    spec = MODELS[keys.pop()]
    recs = [r for r in recs if r["turns_completed"] > 0]
    if args.limit:
        recs = recs[:args.limit]
    out_path = args.out or args.replay.with_suffix(".proj.jsonl")
    if out_path.exists() and not (args.resume or args.check):
        ap.error(f"{out_path} exists; pass --resume or another --out")
    done = {(r["conv_id"], r["lang"]) for r in read_jsonl(out_path)} if args.resume and out_path.exists() else set()
    todo = [r for r in recs if (r["conv_id"], r["lang"]) not in done]
    # similar lengths together keeps padding low
    todo.sort(key=lambda r: sum(len(m["content"]) for m in r["messages"]))
    print(f"{len(recs)} conversations, {len(done)} already projected, {len(todo)} to do -> {out_path}", flush=True)

    t0 = time.time()
    target = spec["target_layer"]
    layers = sorted(set(range(0, spec["n_layers"], args.layer_step)) | {target})
    axis_path = hf_hub_download(AXIS_REPO, f"{spec['axis_folder']}/assistant_axis.pt", repo_type="dataset")
    axis = load_axis(axis_path)
    ref = axis_reference(spec["axis_folder"], target, axis=axis)
    pm = ProbingModel(spec["name"], dtype=torch.bfloat16)  # never with capping hooks
    revision = getattr(pm.model.config, "_commit_hash", None)
    for r in recs:
        if r["model_revision"] not in (revision, None):
            print(f"!! {r['conv_id']}/{r['lang']}: replayed with revision {r['model_revision']}, projecting with {revision}")
            break
    encoder = ConversationEncoder(pm.tokenizer, spec["name"])
    span_mapper = SpanMapper(pm.tokenizer)
    extractor = ActivationExtractor(pm, encoder)
    chat_kwargs = spec["chat_kwargs"]
    print(f"Loaded {spec['name']} rev {revision} in {time.time() - t0:.1f}s; target layer {target}; "
          f"layers {layers}", flush=True)

    if args.check:
        rec = recs[0]
        conv = rec["messages"]
        single = span_mapper.mean_all_turn_activations(pm, encoder, conv, layer=target, **chat_kwargs)
        (batched, spans), = per_turn_activations(extractor, encoder, span_mapper, [conv], layers,
                                                 spec["max_context"], chat_kwargs)
        a = [project(single[i], axis, layer=target) for i, s in enumerate(spans) if s["role"] == "assistant"]
        b = [project(batched[i][layers.index(target)], axis, layer=target) for i, s in enumerate(spans)
             if s["role"] == "assistant"]
        print(f"check {rec['conv_id']}/{rec['lang']}: single {[round(x, 2) for x in a]}")
        print(f"check {rec['conv_id']}/{rec['lang']}: batched {[round(x, 2) for x in b]}")
        print(f"max |single - batched| = {max(abs(x - y) for x, y in zip(a, b)):.4f} "
              f"(normalised-score units: {max(abs(x - y) for x, y in zip(a, b)) / (ref['default_projection'] - ref['role_mean_projection']):.4f})")
        return

    common = {
        "target_layer": target, "axis_file": f"{AXIS_REPO}/{spec['axis_folder']}/assistant_axis.pt",
        "projection_model_revision": revision, "projection_capping": None, "dtype": "bfloat16",
        "default_projection": ref["default_projection"], "role_mean_projection": ref["role_mean_projection"],
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                     "assistant_axis": os.path.dirname(assistant_axis.__file__),
                     "python": platform.python_version(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "created": datetime.now(timezone.utc).isoformat(),
    }

    bs, i, n_rows = args.batch_size, 0, 0
    while i < len(todo):
        batch = todo[i:i + bs]
        t1 = time.time()
        try:
            res = per_turn_activations(extractor, encoder, span_mapper, [r["messages"] for r in batch], layers,
                                       spec["max_context"], chat_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  OOM; batch size now {bs}", flush=True)
            continue
        except RuntimeError as e:
            # find the failing conversation(s): run this batch one at a time, log and skip failures
            print(f"  batch failed ({e}); retrying one by one", flush=True)
            for rec in batch:
                try:
                    (acts, spans), = per_turn_activations(extractor, encoder, span_mapper, [rec["messages"]],
                                                          layers, spec["max_context"], chat_kwargs)
                    rows = rows_for(rec, acts, spans, axis, layers, target, ref, common)
                    if len(rows) != rec["turns_completed"]:
                        raise RuntimeError(f"{len(rows)} projected replies, {rec['turns_completed']} generated")
                except RuntimeError as e1:
                    if isinstance(e1, torch.cuda.OutOfMemoryError):
                        raise
                    log_error(out_path, rec, e1)
                    continue
                for row in rows:
                    append_jsonl(out_path, row)
                n_rows += len(rows)
            i += len(batch)
            continue
        for rec, (acts, spans) in zip(batch, res):
            rows = rows_for(rec, acts, spans, axis, layers, target, ref, common)
            if len(rows) != rec["turns_completed"]:
                log_error(out_path, rec, f"{len(rows)} projected replies, {rec['turns_completed']} generated")
                continue
            for row in rows:
                append_jsonl(out_path, row)
            n_rows += len(rows)
        i += len(batch)
        torch.cuda.empty_cache()
        print(f"{i}/{len(todo)} conversations, {n_rows} turns, batch {len(batch)} in {time.time() - t1:.1f}s",
              flush=True)

    print(f"Done in {time.time() - t0:.1f}s -> {out_path}")
    del pm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
