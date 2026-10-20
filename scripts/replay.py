"""Replay the fixed user turns and generate fresh model replies, turn by turn, batched.

For every (conversation, language) in data/source/conversations.jsonl: append user turn 1,
generate a reply, append it, append user turn 2, ... No system prompt and no instruction
about the reply language: whatever language the model answers in is data.

At turn t, every still-active conversation (all languages and domains) is generated in one
batch. Before each turn, a conversation whose prompt + max_new_tokens would exceed the model
context stops there (stopped_at_turn = the turn it could not run).

Backends:
  vllm  uncapped generation (bfloat16); prompts passed as token ids
  hf    Hugging Face generate, left padding, batched by a token budget; the only backend for
        --capped (assistant_axis build_capping_steerer, published experiment), and the fallback
        when vLLM fails to load (--backend auto)
  stub  --dry-run: no model, replies "stub reply"; exercises the loop, batching, context stop,
        resume and output format on a laptop

Output: <out>/<model>_<capped|uncapped>.jsonl, one line per finished conversation, plus
<model>_<...>.log.jsonl with per-turn throughput.

Usage:
  uv run --project /workspace/assistant-axis python scripts/replay.py --model qwen --limit 2 --out /workspace/outputs/pilot
  python scripts/replay.py --dry-run --model gemma --limit 2 --out <dir>      # laptop, no GPU
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_config import CONVERSATIONS, DEFAULT_OUT, DOMAINS, LANGS, MODELS, append_jsonl, read_jsonl, replay_stem


def request_seed(seed, conv_id, lang, turn):
    """Per-request sampling seed, stable across batching and --resume."""
    h = hashlib.sha256(f"{seed}|{conv_id}|{lang}|{turn}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


def chat_ids(tokenizer, messages, chat_kwargs):
    """Token ids of the chat-formatted prompt, ending with the assistant generation prompt.

    Formats to text then tokenizes without extra special tokens, so Gemma's <bos> appears once
    (the template already writes it) on every transformers version.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def resolve_local(model_name):
    """Local snapshot path and its commit hash (the model revision); downloads if missing."""
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
    try:
        path = snapshot_download(model_name, allow_patterns=patterns, local_files_only=True)
    except Exception:
        path = snapshot_download(model_name, allow_patterns=patterns)
    return path, Path(path).name


def library_versions():
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "vllm"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return v


class StubBackend:
    """No model. One 'token' per 4 characters of a simple chat format; replies 'stub reply'."""

    name = "stub"
    revision = "stub"

    def __init__(self, args, spec):
        self.max_new_tokens = args.max_new_tokens
        self.calls = []  # batch sizes, for tests

    def prompt_ids(self, messages):
        text = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"
        return list(range(max(1, len(text) // 4)))

    def generate(self, prompts, seeds):
        self.calls.append(len(prompts))
        return [("stub reply", 2, "stop") for _ in prompts]

    def close(self):
        pass


class VLLMBackend:
    name = "vllm"

    def __init__(self, args, spec):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        path, self.revision = resolve_local(spec["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.chat_kwargs = spec["chat_kwargs"]
        self.args = args
        self.llm = LLM(model=path, tokenizer=path, dtype="bfloat16", max_model_len=args.max_context,
                       gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                       enable_prefix_caching=True, kv_cache_dtype=args.kv_cache_dtype)
        self.stop_ids = stop_token_ids(self.tokenizer, spec)

    def prompt_ids(self, messages):
        return chat_ids(self.tokenizer, messages, self.chat_kwargs)

    def generate(self, prompts, seeds):
        a = self.args
        params = [self.SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=-1, min_p=0.0,
                                      repetition_penalty=1.0, max_tokens=a.max_new_tokens, seed=s,
                                      stop_token_ids=self.stop_ids, skip_special_tokens=True)
                  for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return [(o.outputs[0].text.strip(), len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]

    def close(self):
        del self.llm
        free_gpu()


class HFBackend:
    """Batched Hugging Face generate with left padding; optional activation capping."""

    name = "hf"

    def __init__(self, args, spec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        path, self.revision = resolve_local(spec["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if "gemma" in spec["name"]:
            kwargs["attn_implementation"] = "eager"  # Gemma 2 logit soft-capping
        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
        self.chat_kwargs = spec["chat_kwargs"]
        self.args = args
        self.stop_ids = stop_token_ids(self.tokenizer, spec)
        self.token_budget = args.hf_token_budget
        self.steerer = None
        if args.capped:
            from huggingface_hub import hf_hub_download
            from assistant_axis import build_capping_steerer, load_capping_config

            cfg_path = hf_hub_download("lu-christina/assistant-axis-vectors", spec["capping_config"], repo_type="dataset")
            self.steerer = build_capping_steerer(self.model, load_capping_config(cfg_path), spec["capping_experiment"])
            self.name = "hf+capping"

    def prompt_ids(self, messages):
        return chat_ids(self.tokenizer, messages, self.chat_kwargs)

    def _chunks(self, order, prompts):
        """Split indices (sorted by length) into sub-batches whose padded size fits the token budget."""
        chunk, longest = [], 0
        for i in order:
            n = len(prompts[i]) + self.args.max_new_tokens
            if chunk and max(longest, n) * (len(chunk) + 1) > self.token_budget:
                yield chunk
                chunk, longest = [], 0
            chunk.append(i)
            longest = max(longest, n)
        if chunk:
            yield chunk

    def _generate_chunk(self, prompts, seed):
        torch, a = self.torch, self.args
        pad = self.tokenizer.pad_token_id
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=self.model.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=self.model.device)
        torch.manual_seed(seed)
        gen_kwargs = dict(input_ids=ids, attention_mask=mask, do_sample=True, temperature=a.temperature,
                          top_p=a.top_p, top_k=0, repetition_penalty=1.0, max_new_tokens=a.max_new_tokens,
                          eos_token_id=self.stop_ids, pad_token_id=pad)
        with torch.inference_mode():
            if self.steerer is not None:
                with self.steerer:
                    out = self.model.generate(**gen_kwargs)
            else:
                out = self.model.generate(**gen_kwargs)
        results = []
        for row in out[:, width:].tolist():
            n, finish = len(row), "length"
            for j, t in enumerate(row):
                if t in self.stop_ids:
                    n, finish = j, "stop"
                    break
            results.append((self.tokenizer.decode(row[:n], skip_special_tokens=True).strip(), n, finish))
        return results

    def generate(self, prompts, seeds):
        # one torch seed per sub-batch (per-row seeds are not possible in a batched HF generate)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results = [None] * len(prompts)
        pending = list(self._chunks(order, prompts))
        while pending:
            chunk = pending.pop(0)
            try:
                outs = self._generate_chunk([prompts[i] for i in chunk], seeds[chunk[0]])
            except self.torch.cuda.OutOfMemoryError:
                free_gpu()
                if len(chunk) == 1:
                    raise
                self.token_budget = max(1, self.token_budget // 2)
                print(f"  OOM on a sub-batch of {len(chunk)}; token budget now {self.token_budget}", flush=True)
                half = len(chunk) // 2
                pending[:0] = [chunk[:half], chunk[half:]]
                continue
            for i, o in zip(chunk, outs):
                results[i] = o
        return results

    def close(self):
        del self.model
        free_gpu()


def stop_token_ids(tokenizer, spec):
    """End-of-turn ids: eos plus <end_of_turn> (Gemma) / <|im_end|> (Qwen)."""
    ids = {tokenizer.eos_token_id}
    for tok in ("<end_of_turn>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def make_backend(args, spec):
    if args.dry_run:
        return StubBackend(args, spec)
    if args.capped or args.backend == "hf":
        return HFBackend(args, spec)
    try:
        return VLLMBackend(args, spec)
    except Exception as e:
        if args.backend == "vllm":
            raise
        print(f"!! vLLM failed to load ({type(e).__name__}: {e}); falling back to Hugging Face generate", flush=True)
        free_gpu()
        return HFBackend(args, spec)


def spread_pick(items, k):
    """k items evenly spaced over the sorted list (covers every persona), deterministic."""
    n = len(items)
    if k >= n:
        return list(items)
    if k == 1:
        return [items[0]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def select_conversations(path, langs, domains, limit, spread=False):
    rows = [r for r in read_jsonl(path) if r["lang"] in langs and r["domain"] in domains]
    keep = set()
    for d in domains:
        ids = sorted({(r["persona_id"], r["topic_id"], r["conv_id"]) for r in rows if r["domain"] == d})
        if limit:
            ids = spread_pick(ids, limit) if spread else ids[:limit]
        keep.update(c for _, _, c in ids)
    rows = [r for r in rows if r["conv_id"] in keep]
    for cid in keep:  # every language runs exactly the same conversations
        got = {r["lang"] for r in rows if r["conv_id"] == cid}
        assert got == set(langs), f"{cid}: languages {sorted(got)} != {sorted(langs)}"
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--limit", type=int, default=0, help="conversations per domain (0 = all)")
    ap.add_argument("--spread", action="store_true",
                    help="with --limit: pick conversations evenly spread across personas/topics "
                         "instead of the first N (use for the capped subset)")
    ap.add_argument("--capped", action="store_true", help="activation capping (Qwen only, HF backend)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume", action="store_true", help="skip (conv_id, lang) pairs already in the output")
    ap.add_argument("--backend", choices=("auto", "vllm", "hf"), default="auto")
    ap.add_argument("--max-context", type=int, help="override the model context limit (tests)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="vLLM KV cache dtype; 'fp8' roughly doubles how many conversations fit at once "
                         "(generation only; projection always reloads the bf16 model)")
    ap.add_argument("--hf-token-budget", type=int, default=60000,
                    help="HF backend: max (padded prompt + new tokens) x batch rows per sub-batch")
    ap.add_argument("--conversations", type=Path, default=CONVERSATIONS)
    ap.add_argument("--dry-run", action="store_true", help="stub generator, no model or GPU")
    ap.add_argument("--stop-after-turn", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    spec = MODELS[args.model]
    if args.capped and not spec["capping_experiment"]:
        ap.error(f"{args.model} has no published capping setting; --capped is Qwen only")
    args.max_context = args.max_context or spec["max_context"]
    langs, domains = args.langs.split(","), args.domains.split(",")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = replay_stem(args.model, args.capped)
    out_path, log_path = args.out / f"{stem}.jsonl", args.out / f"{stem}.log.jsonl"
    ckpt_path = args.out / f"{stem}.ckpt.json"
    if (out_path.exists() or ckpt_path.exists()) and not args.resume:
        ap.error(f"{out_path} or its checkpoint exists; pass --resume to continue it or choose another --out")
    fingerprint = {"seed": args.seed, "capped": args.capped, "temperature": args.temperature, "top_p": args.top_p,
                   "max_new_tokens": args.max_new_tokens, "max_context": args.max_context}

    convs = select_conversations(args.conversations, langs, domains, args.limit, args.spread)
    done = set()
    if args.resume and out_path.exists():
        prev = read_jsonl(out_path)
        done = {(r["conv_id"], r["lang"]) for r in prev}
        changed = {k for r in prev for k, v in (("seed", args.seed), ("capped", args.capped)) if r[k] != v}
        changed |= {k for r in prev for k, v in r["sampling"].items() if getattr(args, k) != v}
        if changed:
            ap.error(f"--resume with different settings than the existing file: {sorted(changed)}")
    todo = [c for c in convs if (c["conv_id"], c["lang"]) not in done]
    ckpt = None
    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        diff = sorted(k for k, v in fingerprint.items() if ckpt["settings"].get(k) != v)
        if diff:
            ap.error(f"--resume: checkpoint was written with different settings: {diff}")
        ck_keys = {(v["conv_id"], v["lang"]) for v in ckpt["states"]}
        todo_keys = {(c["conv_id"], c["lang"]) for c in todo}
        if ck_keys != todo_keys:
            ap.error(f"--resume: checkpoint holds {len(ck_keys)} unfinished conversations but this selection has "
                     f"{len(todo_keys)} left to run ({len(ck_keys ^ todo_keys)} differ); use the same "
                     f"--langs/--domains/--limit/--spread as the interrupted run")
        print(f"Resuming from checkpoint: turns 1-{ckpt['next_t']} done for {len(ck_keys)} conversations", flush=True)
    print(f"{len(convs)} (conversation, language) pairs selected; {len(done & {(c['conv_id'], c['lang']) for c in convs})} "
          f"already done; {len(todo)} to run -> {out_path}", flush=True)
    if not todo:
        return out_path

    t0 = time.time()
    backend = make_backend(args, spec)
    versions = library_versions()
    print(f"Backend {backend.name} ready in {time.time() - t0:.1f}s; model {spec['name']} rev {backend.revision}; "
          f"context {args.max_context}", flush=True)

    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    common = {
        "model": spec["name"], "model_key": args.model, "model_revision": backend.revision, "backend": backend.name,
        "capped": args.capped, "capping_experiment": spec["capping_experiment"] if args.capped else None,
        "seed": args.seed,
        "seed_scheme": ("per request: sha256(seed|conv_id|lang|turn)" if backend.name in ("vllm", "stub")
                        else "per HF sub-batch: torch.manual_seed(request seed of the sub-batch's first row)"),
        "sampling": sampling, "sampling_fixed": {"top_k": "off", "min_p": 0.0, "repetition_penalty": 1.0},
        "chat_kwargs": spec["chat_kwargs"], "system_prompt": None, "max_context": args.max_context,
        "kv_cache_dtype": args.kv_cache_dtype if backend.name == "vllm" else "model dtype (bf16)",
        "gpu_memory_utilization": args.gpu_memory_utilization if backend.name == "vllm" else None,
        "versions": versions,
    }
    state = [{"conv": c, "messages": [], "prompt_tokens": [], "reply_tokens": [], "finish_reasons": [],
              "request_seeds": [], "stopped_at_turn": None, "start": None, "done": False} for c in todo]
    start_t = 0
    if ckpt:
        saved = {(v["conv_id"], v["lang"]): v for v in ckpt["states"]}
        for s in state:
            v = saved[(s["conv"]["conv_id"], s["conv"]["lang"])]
            for k in ("messages", "prompt_tokens", "reply_tokens", "finish_reasons", "request_seeds", "stopped_at_turn"):
                s[k] = v[k]
            s["start"] = time.time() - v["elapsed"] if v["elapsed"] is not None else None
        start_t = ckpt["next_t"]

    def save_checkpoint(next_t):
        """All unfinished conversations after a complete turn; atomic replace, so a crash loses at most one turn."""
        rows = [{"conv_id": s["conv"]["conv_id"], "lang": s["conv"]["lang"],
                 **{k: s[k] for k in ("messages", "prompt_tokens", "reply_tokens", "finish_reasons",
                                      "request_seeds", "stopped_at_turn")},
                 "elapsed": (time.time() - s["start"]) if s["start"] else None}
                for s in state if not s["done"]]
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"settings": fingerprint, "next_t": next_t, "states": rows,
                                   "saved": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, ckpt_path)
    n_turns = max(len(c["user_turns"]) for c in todo)
    n_written = 0

    def finish(s):
        nonlocal n_written
        c = s["conv"]
        row = {"conv_id": c["conv_id"], "domain": c["domain"], "lang": c["lang"], "persona_id": c["persona_id"],
               "topic_id": c["topic_id"], **common,
               "messages": s["messages"], "n_user_turns": len(c["user_turns"]),
               "turns_completed": len(s["reply_tokens"]), "stopped_at_turn": s["stopped_at_turn"],
               "prompt_tokens": s["prompt_tokens"], "reply_tokens": s["reply_tokens"],
               "finish_reasons": s["finish_reasons"], "request_seeds": s["request_seeds"],
               "wall_seconds": round(time.time() - s["start"], 1),
               "created": datetime.now(timezone.utc).isoformat(), "source_file": c["source_file"]}
        append_jsonl(out_path, row)
        s["done"] = True
        n_written += 1

    for t in range(start_t, n_turns):
        if args.stop_after_turn and t >= args.stop_after_turn:
            print(f"Stopping after turn {t} (--stop-after-turn, for tests)", flush=True)
            backend.close()
            return out_path
        turn = t + 1
        active, prompts = [], []
        for s in state:
            c = s["conv"]
            if s["stopped_at_turn"] is not None or t >= len(c["user_turns"]):
                continue
            if s["start"] is None:
                s["start"] = time.time()
            msgs = s["messages"] + [{"role": "user", "content": c["user_turns"][t]}]
            ids = backend.prompt_ids(msgs)
            if len(ids) + args.max_new_tokens > args.max_context:
                s["stopped_at_turn"] = turn
                print(f"  {c['conv_id']}/{c['lang']}: context stop at turn {turn} "
                      f"({len(ids)} prompt + {args.max_new_tokens} > {args.max_context})", flush=True)
                finish(s)
                continue
            active.append(s)
            prompts.append(ids)
        if not active:
            break
        seeds = [request_seed(args.seed, s["conv"]["conv_id"], s["conv"]["lang"], turn) for s in active]
        t1 = time.time()
        outs = backend.generate(prompts, seeds)
        dt = time.time() - t1
        n_new = sum(o[1] for o in outs)
        for s, ids, seed, (text, n, fin) in zip(active, prompts, seeds, outs):
            s["messages"] += [{"role": "user", "content": s["conv"]["user_turns"][t]},
                              {"role": "assistant", "content": text}]
            s["prompt_tokens"].append(len(ids))
            s["reply_tokens"].append(n)
            s["finish_reasons"].append(fin)
            s["request_seeds"].append(seed)
            if len(s["reply_tokens"]) == len(s["conv"]["user_turns"]):
                finish(s)
        n_len = sum(o[2] == "length" for o in outs)
        log = {"turn": turn, "active": len(active), "new_tokens": n_new, "prompt_tokens": sum(map(len, prompts)),
               "seconds": round(dt, 2), "tokens_per_s": round(n_new / dt, 1) if dt > 0 else None,
               "hit_max_new_tokens": n_len, "backend": backend.name}
        append_jsonl(log_path, log)
        print(f"turn {turn:2d}: {len(active):4d} active, {n_new:7d} new tokens in {dt:7.1f}s "
              f"({log['tokens_per_s']} tok/s), {n_len} hit max_new_tokens, {n_written} conversations written",
              flush=True)
        save_checkpoint(turn)

    backend.close()
    if ckpt_path.exists():
        os.replace(ckpt_path, ckpt_path.with_suffix(".done.json"))
    print(f"Done: {n_written} conversations written to {out_path} in {time.time() - t0:.1f}s", flush=True)
    return out_path


if __name__ == "__main__":
    main()
