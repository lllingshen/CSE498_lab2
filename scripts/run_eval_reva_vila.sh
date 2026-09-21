#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV=${CONDA_ENV-"vila"}
PYTHON=(python3)
if [ -n "$CONDA_ENV" ]; then
    PYTHON=(conda run -n "$CONDA_ENV" python3)
fi
VILA_REPO=${VILA_REPO:-"<VILA_REPO>"}
MODEL_PATH=${MODEL_PATH:-"Efficient-Large-Model/VILA1.5-3b"}
MODEL_BASE=${MODEL_BASE:-""}
REVA_ROOT=${REVA_ROOT:-"$PROJECT_ROOT/data/reva_test"}
REVA_JSON=${REVA_JSON:-"$REVA_ROOT/test_set.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/vila_reva_v2"}
MAX_QUESTIONS=${MAX_QUESTIONS:-""}
NUM_VIDEO_FRAMES=${NUM_VIDEO_FRAMES:-4}
VIDEO_MAX_TILES=${VIDEO_MAX_TILES:-""}
GPU=${GPU:-""}

if [ ! -d "$VILA_REPO" ]; then
  echo "VILA_REPO does not exist: $VILA_REPO" >&2
  echo "Set VILA_REPO=/path/to/VILA before running this script." >&2
  exit 1
fi

if [ ! -f "$REVA_JSON" ]; then
  echo "ReVA annotation file does not exist: $REVA_JSON" >&2
  exit 1
fi

args=(
  "$PROJECT_ROOT/vila_eval/reva_v2.py"
  --model-path "$MODEL_PATH"
  --question-file "$REVA_JSON"
  --dataset-root "$REVA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --num-video-frames "$NUM_VIDEO_FRAMES"
)

if [ -n "$MODEL_BASE" ]; then
  args+=(--model-base "$MODEL_BASE")
fi

if [ -n "$MAX_QUESTIONS" ]; then
  args+=(--max-questions "$MAX_QUESTIONS")
fi

if [ -n "$VIDEO_MAX_TILES" ]; then
  args+=(--video-max-tiles "$VIDEO_MAX_TILES")
fi

if [ -n "$GPU" ]; then
  args+=(--gpu "$GPU")
fi

echo "=== VILA ReVA_V2 Evaluation ==="
echo "VILA repo:  $VILA_REPO"
echo "Model:      $MODEL_PATH"
echo "ReVA json:  $REVA_JSON"
echo "ReVA root:  $REVA_ROOT"
echo "Output:     $OUTPUT_DIR"

PYTHONPATH="$VILA_REPO:${PYTHONPATH:-}" "${PYTHON[@]}" "${args[@]}"
