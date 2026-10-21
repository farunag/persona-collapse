#!/bin/bash
cd /workspace/persona-collapse/scripts
P="uv run --project /workspace/assistant-axis"
V=/workspace/vllm-env/bin/python
R=/workspace/outputs/replay
$V replay.py --model qwen --max-new-tokens 1024 --out $R > /workspace/outputs/logs/full_uncapped.log 2>&1 \
  && $P python replay.py --model qwen --capped --domains therapy,metacognitive --limit 10 --spread \
       --max-new-tokens 1024 --hf-token-budget 65000 --out $R > /workspace/outputs/logs/full_capped.log 2>&1
echo "block3 exit $?" >> /workspace/outputs/logs/full_uncapped.log
