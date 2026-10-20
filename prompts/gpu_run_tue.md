# GPU run, Tue 20 Oct — Qwen 3 32B (uncapped + capped)

Run these on the Vast server, in tmux, one block at a time. Stop and report if any
block prints an error. P = the assistant-axis env; V = the separate vLLM env.

## 0. New instance setup (~15-20 min)
    tmux new -s main
    cd /workspace && git clone https://github.com/farunag/persona-collapse.git
    export HF_TOKEN=hf_xxx
    bash /workspace/persona-collapse/scripts/setup.sh
    source ~/.bashrc
    gh auth login                      # GitHub.com, HTTPS, Yes, web browser
    git config --global user.name "farunag"; git config --global user.email "farunagodwin01@gmail.com"

## 1. Environments + check (~10 min)
    cd /workspace/persona-collapse && git pull
    P="uv run --project /workspace/assistant-axis"
    V=/workspace/vllm-env/bin/python
    uv pip install --python /workspace/assistant-axis/.venv/bin/python pytest matplotlib fasttext-wheel
    bash scripts/setup_vllm.sh         # separate env; must print "cuda ok True" and the GPU name
    $P python scripts/check_env.py     # vllm FAIL is expected here (it lives in its own env)
    $P python -m pytest -q tests

## 2. Pilot: 2 conversations per domain x 3 languages (~10 min)
    cd scripts
    $V replay.py --model qwen --limit 2 --out /workspace/outputs/pilot
    $P python replay.py --model qwen --limit 2 --capped --out /workspace/outputs/pilot
    $P python project.py --replay /workspace/outputs/pilot/qwen_uncapped.jsonl --check
    $P python project.py --replay /workspace/outputs/pilot/qwen_uncapped.jsonl
    $P python project.py --replay /workspace/outputs/pilot/qwen_capped.jsonl
    $P python langid.py  --replay /workspace/outputs/pilot/qwen_uncapped.jsonl
    $P python langid.py  --replay /workspace/outputs/pilot/qwen_capped.jsonl
    $P python analyze.py --min-convs 2 --out /workspace/outputs/pilot/results \
       --proj /workspace/outputs/pilot/qwen_{uncapped,capped}.proj.jsonl \
       --langid /workspace/outputs/pilot/qwen_{uncapped,capped}.langid.jsonl \
       --replay /workspace/outputs/pilot/qwen_{uncapped,capped}.jsonl
  Check before going on: the replay log says "Backend vllm" (not a fall-back); the --check
  difference is tiny; a few Hausa and Yoruba replies read sensibly; tokens/s per turn.
  Send the numbers to Claude (chat) before the full run.

## 3. Full Qwen run (uncapped with vLLM, then capped; run them one after the other)
  The pilot hit the 400-token reply cap on ~99% of replies, so both runs use
  --max-new-tokens 1024. The capped run uses --hf-token-budget 65000 (the pilot peaked
  near 90 GiB at 60000) and --limit 10 per domain.
    R=/workspace/outputs/replay
    $V replay.py --model qwen --max-new-tokens 1024 --out $R
    $P python replay.py --model qwen --capped --domains therapy,metacognitive --limit 10 --spread \
       --max-new-tokens 1024 --hf-token-budget 65000 --out $R
  If interrupted: rerun the same line with --resume.

## 4. Projection, language ID, analysis
    $P python project.py --replay $R/qwen_uncapped.jsonl
    $P python project.py --replay $R/qwen_capped.jsonl
    $P python langid.py  --replay $R/qwen_uncapped.jsonl
    $P python langid.py  --replay $R/qwen_capped.jsonl
    $P python analyze.py --proj $R/qwen_{uncapped,capped}.proj.jsonl \
       --langid $R/qwen_{uncapped,capped}.langid.jsonl --replay $R/qwen_{uncapped,capped}.jsonl
    cd .. && git add results && git commit -m "Stage 2: Qwen results" && git push

## 5. Keep the raw outputs safe before stopping the instance
    tar czf /workspace/qwen_outputs.tgz -C /workspace outputs
  (They go to the Hugging Face Dataset on Thursday; until then do not destroy the
  instance without copying this file somewhere, e.g. scp to your laptop.)
