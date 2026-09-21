#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EVAL_NAME=${EVAL_NAME:-qwen_base} \
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs"} \
bash "$PROJECT_ROOT/scripts/run_eval_reva.sh"
