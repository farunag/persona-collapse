#!/bin/bash
set -x
export HF_HOME=/workspace/hf_cache
cd /workspace/persona-collapse
P="uv run --project /workspace/assistant-axis"
R=/workspace/outputs/replay
$P python scripts/project.py --replay $R/qwen_capped.jsonl --check || exit 1
$P python scripts/project.py --replay $R/qwen_capped.jsonl || exit 2
$P python scripts/langid.py --replay $R/qwen_capped.jsonl || exit 3
echo "capped post exit 0 $(date -u +%FT%TZ)"
