#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-model_config.yaml>"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/mnt/core_llm_large/shruthan/debug/fastllm_training_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# pip install transformers==5.0.0rc0
cd /mnt/core_llm2/shruthan/git/Fast-LLM
pip install -e . --no-build-isolation

python -m torch.distributed.run --nproc_per_node=8 --redirects=3 --log_dir="$LOG_DIR" fast_llm/tools/train.py gpt --config "$1"