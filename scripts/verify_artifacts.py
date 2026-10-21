"""Check that every run output on this machine is saved in the Hugging Face dataset (same size),
and that the files the project needs exist there. Run before destroying an instance.

  HF_TOKEN="$(cat /root/.hf_write_token)" uv run --project /workspace/assistant-axis \
      python scripts/verify_artifacts.py --machine gemma            # or --machine qwen
  add --upload-missing to upload anything missing or of a different size, then re-check.

Local -> dataset mapping (the layout in DATA.md):
  /workspace/outputs/replay/<model>_*      -> <model>/replay/
  /workspace/outputs/pilot/                -> qwen/pilot/
  /workspace/outputs/pilot_gemma/          -> gemma/pilot/
  /workspace/outputs/logs/                 -> <machine>/logs/
"""

import argparse
import sys
from pathlib import Path

REPO = "Faruna01/persona-collapse-runs"
OUT = Path("/workspace/outputs")
REQUIRED = {
    "qwen": [f"qwen/replay/qwen_{c}{s}" for c in ("uncapped", "capped")
             for s in (".jsonl", ".log.jsonl", ".proj.jsonl", ".langid.jsonl")],
    "gemma": [f"gemma/replay/gemma_uncapped{s}" for s in (".jsonl", ".log.jsonl", ".proj.jsonl", ".langid.jsonl")],
}


def local_files(machine):
    pairs = []
    rep = OUT / "replay"
    if rep.is_dir():
        for f in sorted(rep.iterdir()):
            if f.is_file() and f.name.split("_")[0] in ("qwen", "gemma") and not f.name.endswith((".tmp", ".ckpt.json")):
                pairs.append((f, f"{f.name.split('_')[0]}/replay/{f.name}"))
    for d, prefix in ((OUT / "pilot", "qwen/pilot"), (OUT / "pilot_gemma", "gemma/pilot"), (OUT / "logs", f"{machine}/logs")):
        if d.is_dir():
            pairs += [(f, f"{prefix}/{f.relative_to(d).as_posix()}") for f in sorted(d.rglob("*")) if f.is_file()]
    return pairs


def remote_sizes(api):
    return {e.path: e.size for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
            if getattr(e, "size", None) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=("qwen", "gemma"), required=True)
    ap.add_argument("--upload-missing", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    remote = remote_sizes(api)
    bad = []
    for f, path in local_files(a.machine):
        size = f.stat().st_size
        state = "ok" if remote.get(path) == size else ("MISSING" if path not in remote else f"SIZE {remote[path]} != {size}")
        if state != "ok":
            bad.append((f, path))
        print(f"{state:>8}  {path}  ({size / 1e6:.1f} MB)")
    if bad and a.upload_missing:
        for f, path in bad:
            print(f"uploading {path}")
            api.upload_file(path_or_fileobj=str(f), path_in_repo=path, repo_id=REPO, repo_type="dataset")
        remote, bad = remote_sizes(api), [(f, p) for f, p in bad if remote_sizes(api).get(p) != f.stat().st_size]
    missing_req = [p for p in REQUIRED[a.machine] if p not in remote]
    print("\nRequired for this machine's runs:")
    for p in REQUIRED[a.machine]:
        print(f"  {'ok' if p in remote else 'MISSING':>7}  {p}")
    if bad or missing_req:
        print(f"\nNOT SAFE TO DESTROY: {len(bad)} local files not saved, {len(missing_req)} required files missing")
        sys.exit(1)
    print("\nALL REQUIRED FILES SAVED")


if __name__ == "__main__":
    main()
