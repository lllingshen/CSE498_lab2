#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${MODEL_PATH:-}" ]; then
  echo "Set MODEL_PATH to the fine-tuned checkpoint directory before running this script." >&2
  echo "Example: MODEL_PATH=$PROJECT_ROOT/outputs/qwen_reva_sft/checkpoint-2 bash scripts/run_eval_qwen_finetuned.sh" >&2
  exit 1
fi

if [ -n "${MODEL_BASE:-}" ]; then
  if [ ! -f "$MODEL_PATH/adapter_config.json" ] || { [ ! -s "$MODEL_PATH/adapter_model.safetensors" ] && [ ! -s "$MODEL_PATH/adapter_model.bin" ]; }; then
    echo "Invalid LoRA directory: $MODEL_PATH. Expected adapter_config.json and nonempty adapter_model.safetensors or adapter_model.bin." >&2
    exit 1
  fi
fi

if [ -d "$MODEL_PATH" ]; then
  export MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
fi
if [ -n "${MODEL_BASE:-}" ] && [ -d "$MODEL_BASE" ]; then
  export MODEL_BASE="$(cd "$MODEL_BASE" && pwd)"
fi

EVAL_NAME=${EVAL_NAME:-qwen_sft} \
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs"} \
bash "$PROJECT_ROOT/scripts/run_eval_reva.sh"
