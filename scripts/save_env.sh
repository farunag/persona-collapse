#!/usr/bin/env bash
# Record this machine's environment for replication into env/<name>/ and commit it.
#   bash scripts/save_env.sh qwen-rtxpro6000      (or gemma-a100)
set -uo pipefail
NAME=${1:?usage: save_env.sh <machine-name>}
ROOT=$(cd "$(dirname "$0")/.." && pwd); D="$ROOT/env/$NAME"; mkdir -p "$D"
nvidia-smi > "$D/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$D/gpu.csv" 2>&1
(nvcc --version 2>/dev/null || echo "nvcc not installed") > "$D/cuda.txt"
uname -a > "$D/uname.txt"
git -C /workspace/assistant-axis rev-parse HEAD > "$D/assistant_axis_commit.txt" 2>&1
git -C "$ROOT" rev-parse HEAD > "$D/repo_commit.txt"
uv pip freeze --python /workspace/assistant-axis/.venv/bin/python > "$D/assistant_axis_env.txt" 2>&1
[ -x /workspace/vllm-env/bin/python ] && uv pip freeze --python /workspace/vllm-env/bin/python > "$D/vllm_env.txt" 2>&1
ls /workspace/outputs/logs/*.sh >/dev/null 2>&1 && cp /workspace/outputs/logs/*.sh "$D/" 2>/dev/null
history 2>/dev/null | tail -n 0 >/dev/null
echo "Saved environment to $D"
cd "$ROOT" && git add "env/$NAME" && git commit -qm "Environment record: $NAME" && git pull -q --rebase && git push -q && echo "pushed env/$NAME"
