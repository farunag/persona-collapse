#!/usr/bin/env bash
# Persona Collapse — one-time setup for a fresh Vast.ai GPU instance.
# Usage (on the server, inside tmux):
#   export HF_TOKEN=hf_xxx          # your Hugging Face token (read access is enough)
#   bash setup.sh
set -euo pipefail

# On Vast: /workspace (default). On Colab: run with  WORK=/content bash setup.sh
WORK=${WORK:-/workspace}
mkdir -p "$WORK" && cd "$WORK"

echo "== 1. System packages =="
apt-get update -y
apt-get install -y tmux git curl htop nvtop jq

echo "== 1b. GitHub CLI (for pushing code) =="
if ! command -v gh >/dev/null; then
  mkdir -p -m 755 /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli-archive-keyring.gpg
  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt-get update -y && apt-get install -y gh
fi

echo "== 2. GPU check =="
nvidia-smi
df -h "$WORK"

echo "== 3. Persistent caches on the big disk =="
cat >> ~/.bashrc <<EOF
export HF_HOME=$WORK/hf_cache
export UV_CACHE_DIR=$WORK/uv_cache
export PATH="\$HOME/.local/bin:\$PATH"
EOF
export HF_HOME=$WORK/hf_cache
export UV_CACHE_DIR=$WORK/uv_cache
export PATH="$HOME/.local/bin:$PATH"

echo "== 4. uv (Python package manager used by assistant-axis) =="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== 5. Claude Code =="
# Native installer from the official docs. If this URL ever fails, follow
# the current install instructions at https://code.claude.com/docs
curl -fsSL https://claude.ai/install.sh | bash || echo "!! Claude Code install failed — install manually from the docs"

echo "== 6. assistant-axis library =="
if [ ! -d "$WORK/assistant-axis" ]; then
  git clone https://github.com/safety-research/assistant-axis.git "$WORK/assistant-axis"
fi
cd "$WORK/assistant-axis"
uv sync
cd "$WORK"

echo "== 7. Download models + published axes (~120 GB, takes a while) =="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN not set. Run: export HF_TOKEN=hf_xxx  then re-run this script."
  exit 1
fi
cd "$WORK/assistant-axis"
uv run python - <<'PY'
from huggingface_hub import snapshot_download
# Published axes, role vectors and capping configs
snapshot_download("lu-christina/assistant-axis-vectors", repo_type="dataset",
                  allow_patterns=["qwen-3-32b/*", "gemma-2-27b/*", "README.md"])
# Models (Gemma is gated: accept its licence on huggingface.co first)
for m in ["Qwen/Qwen3-32B", "google/gemma-2-27b-it"]:
    print("Downloading", m)
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
print("All downloads done.")
PY

echo
echo "== Setup complete =="
echo "Next: run 'claude', type /login, open the link on your laptop/phone, paste the code back."
