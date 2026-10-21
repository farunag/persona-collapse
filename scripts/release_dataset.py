"""Add the flat tables and the dataset card to Faruna01/persona-collapse-runs.

Needs HF_TOKEN (write access) in the environment; never pass it on the command line.
Does NOT change the dataset's visibility: switch it to public on the Hub when ready.

  python scripts/release_dataset.py            # download, export, upload data/ + README.md
  python scripts/release_dataset.py --dry-run  # download and export only, upload nothing
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "Faruna01/persona-collapse-runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("release_work"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN in the environment first (a write token).")

    raw = Path(snapshot_download(REPO, repo_type="dataset", local_dir=a.work / "runs",
                                 allow_patterns=["qwen/replay/*.jsonl", "gemma/replay/*.jsonl", "sensitivity/*"]))
    out = a.work / "export"
    subprocess.run([sys.executable, "scripts/export_dataset.py", "--raw", str(raw), "--out", str(out)], check=True)
    if a.dry_run:
        print("Dry run: nothing uploaded. Tables are in", out / "data")
        return
    api = HfApi()
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(out / "data"), path_in_repo="data",
                      commit_message="Flat tables: turns-*.jsonl and sensitivity.jsonl (scripts/export_dataset.py)")
    api.upload_file(repo_id=REPO, repo_type="dataset", path_or_fileobj="dataset/README.md", path_in_repo="README.md",
                    commit_message="Dataset card")
    print(f"Uploaded. Check https://huggingface.co/datasets/{REPO} (the viewer may take a few minutes).")


if __name__ == "__main__":
    main()
