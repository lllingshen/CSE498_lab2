#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${MODEL_PATH:-}" ]; then
  echo "Set MODEL_PATH to the fine-tuned checkpoint directory before running this script." >&2
  echo "Example: MODEL_PATH=$PROJECT_ROOT/outputs/qwen_reva_sft/checkpoint-2 bash scripts/run_eval_qwen_finetuned.sh" >&2
  exit 1
fi

EVAL_NAME=${EVAL_NAME:-qwen_sft} \
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs"} \
bash "$PROJECT_ROOT/scripts/run_eval_reva.sh"
