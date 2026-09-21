#!/bin/bash
set -e

cd "$(dirname "$0")/.."

python3 scripts/setup_demo_data.py

REVA_TRAIN_JSON=data/demo_reva/train_set.json \
QWEN_TRAIN_JSON=data/qwen_train/train.json \
QWEN_VIDEO_ROOT=data/qwen_train \
MAX_SAMPLES=0 \
REQUIRE_VIDEO=1 \
bash scripts/prepare_qwen_train_data.sh
