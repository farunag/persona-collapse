#!/bin/bash
cd /workspace/persona-collapse
S=/workspace/outputs/sensitivity
echo "start $(date -u +%FT%TZ)"
uv run --project /workspace/assistant-axis python scripts/judge_roles.py --replay $S/qwen_uncapped.jsonl $S/gemma_uncapped.jsonl \
  --out $S/role_judgments.jsonl
echo "judge exit $? $(date -u +%FT%TZ)"
