#!/bin/bash
set -e

cd "$(dirname "$0")/.."
python3 scripts/compare_model_metrics.py "$@"
