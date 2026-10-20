"""Stage 2 GPU setup check: libraries, GPU, HF token, cached models and axes. Downloads nothing.

  uv run --project /workspace/assistant-axis python scripts/check_env.py
"""

import importlib
import os
import shutil

from stage2_config import MODELS


def main():
    ok = True
    for mod in ("torch", "transformers", "vllm", "assistant_axis", "fasttext", "matplotlib", "numpy", "pytest"):
        try:
            m = importlib.import_module(mod)
            print(f"ok   {mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"FAIL {mod}: {type(e).__name__}: {e}")
            ok = ok and mod not in ("torch", "transformers", "assistant_axis")
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"ok   GPU {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, "
              f"{free / 2**30:.0f}/{total / 2**30:.0f} GiB free, CUDA {torch.version.cuda}, bf16 {torch.cuda.is_bf16_supported()}")
    else:
        print("FAIL no CUDA GPU visible")
        ok = False
    print(f"{'ok  ' if os.environ.get('HF_TOKEN') else 'WARN'} HF_TOKEN {'set' if os.environ.get('HF_TOKEN') else 'not set (Gemma is gated)'}")
    print(f"     HF_HOME={os.environ.get('HF_HOME')}; disk free {shutil.disk_usage('/workspace' if os.path.isdir('/workspace') else '.').free / 2**30:.0f} GiB")
    from huggingface_hub import snapshot_download, hf_hub_download
    for key, spec in MODELS.items():
        try:
            p = snapshot_download(spec["name"], local_files_only=True, allow_patterns=["*.json"])
            print(f"ok   {spec['name']} cached, revision {os.path.basename(p)}")
        except Exception:
            print(f"WARN {spec['name']} not in the local cache (replay.py would download it)")
        files = [f"{spec['axis_folder']}/assistant_axis.pt"] + ([spec["capping_config"]] if spec["capping_config"] else [])
        for f in files:
            try:
                hf_hub_download("lu-christina/assistant-axis-vectors", f, repo_type="dataset", local_files_only=True)
                print(f"ok   axis file {f} cached")
            except Exception:
                print(f"WARN axis file {f} not cached")
    print("READY" if ok else "NOT READY")


if __name__ == "__main__":
    main()
