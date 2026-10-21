#!/bin/bash
# Block 3 with fp8 KV cache via vLLM's Triton attention backend. (b) runs only if (a) succeeds.
# Rerunning this script resumes: each step adds --resume once its output or checkpoint exists.
# Times go to block3_times.log.
export HF_HOME=/workspace/hf_cache
cd /workspace/persona-collapse/scripts
P="uv run --project /workspace/assistant-axis"
V=/workspace/vllm-env/bin/python
R=/workspace/outputs/replay
T=/workspace/outputs/logs/block3_times.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
res() { [ -e "$R/$1.jsonl" ] || [ -e "$R/$1.ckpt.json" ] && echo --resume; }
echo "(a) uncapped vLLM fp8-KV TRITON_ATTN start: $(ts) $(res qwen_uncapped)" >> $T
$V replay.py --model qwen --max-new-tokens 1024 --backend vllm --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
   --gpu-memory-utilization 0.95 --out $R $(res qwen_uncapped) >> /workspace/outputs/logs/full_uncapped_fp8.log 2>&1
A=$?
echo "(a) uncapped end:   $(ts) exit $A" >> $T
[ $A -ne 0 ] && { echo "(b) not started because (a) failed" >> $T; exit $A; }
echo "(b) capped HF start: $(ts) $(res qwen_capped)" >> $T
$P python replay.py --model qwen --capped --domains therapy,metacognitive --limit 10 --spread --max-new-tokens 1024 \
   --hf-token-budget 65000 --out $R $(res qwen_capped) >> /workspace/outputs/logs/full_capped.log 2>&1
B=$?
echo "(b) capped HF end:   $(ts) exit $B" >> $T
