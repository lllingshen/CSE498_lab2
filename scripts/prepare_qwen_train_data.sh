#!/bin/bash
set -e

cd "$(dirname "$0")/.."

require_video_args=()
if [ "${REQUIRE_VIDEO:-1}" = "1" ]; then
  require_video_args+=(--require-video)
fi

python3 scripts/build_qwen_train_data.py \
  --input "${REVA_TRAIN_JSON:-data/reva_train/train_set.json}" \
  --output "${QWEN_TRAIN_JSON:-data/qwen_train/train.json}" \
  --video-root "${QWEN_VIDEO_ROOT:-data/qwen_train}" \
  --max-samples "${MAX_SAMPLES:-200}" \
  --answer-style "${ANSWER_STYLE:-tagged}" \
  --prompt-style "${PROMPT_STYLE:-reva_eval}" \
  "${require_video_args[@]}" \
  --keep-metadata
