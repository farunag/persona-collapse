#!/usr/bin/env bash
# Separate Python environment for vLLM, so installing vLLM cannot change the torch /
# transformers versions that assistant-axis (and project.py, capping) depend on.
# replay.py's uncapped run uses this env; everything else uses the assistant-axis env.
#   bash scripts/setup_vllm.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv venv /workspace/vllm-env --python 3.12
uv pip install --python /workspace/vllm-env/bin/python vllm huggingface_hub
/workspace/vllm-env/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
