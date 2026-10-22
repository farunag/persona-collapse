"""Build the workbench data from the public dataset and publish the Hugging Face Space.

Needs HF_TOKEN (write) in the environment only for the upload; never pass it on the command line.

  python scripts/deploy_space.py              # build space/data/ and upload to Faruna01/persona-collapse
  python scripts/deploy_space.py --no-upload  # build only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DATASET = "Faruna01/persona-collapse-runs"
SPACE = "Faruna01/persona-collapse"
LABELS = {"qwen": "Full run: 180 conversations per language, 10 capped per topic",
          "gemma": "Full run: 180 conversations per language, default only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("space_work"))
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    raw = Path(snapshot_download(DATASET, repo_type="dataset", local_dir=a.work / "runs",
                                 allow_patterns=["qwen/replay/*.jsonl", "gemma/replay/*.jsonl"]))
    data = Path("space/data")
    if data.exists():
        shutil.rmtree(data)
    for model in ("qwen", "gemma"):
        subprocess.run([sys.executable, "scripts/build_space_data.py", "--dir", str(raw / model / "replay"),
                        "--model", model, "--split-dir", str(data), "--label", LABELS[model]], check=True)
    if a.no_upload:
        print("Built", data, "- nothing uploaded.")
        return
    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN in the environment to upload.")
    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path="space",
                      allow_patterns=["index.html", "README.md", "data/**"],
                      commit_message="Workbench with full runs (scripts/deploy_space.py)")
    print(f"Published: https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
