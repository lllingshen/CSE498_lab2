#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "${CONDA_ENV:-}" ] && [ -z "${INSIDE_CONDA_RUN:-}" ]; then
  CONDA_BIN=${CONDA_BIN:-conda}
  INSIDE_CONDA_RUN=1 exec "$CONDA_BIN" run -n "$CONDA_ENV" bash "$0"
fi

cd "$PROJECT_ROOT/qwen_finetune"
bash scripts/sft_7b.sh
