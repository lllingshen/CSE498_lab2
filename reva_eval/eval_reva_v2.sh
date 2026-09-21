#!/bin/bash
# Evaluate Qwen-VL models on ReVA_V2 test_set.json.

set -e
cd "$(dirname "$0")"

PROJECT_ROOT="$(cd .. && pwd)"
CONDA_ENV=${CONDA_ENV-"qwen2"}
PYTHON=(python3)
if [ -n "$CONDA_ENV" ]; then
    PYTHON=(conda run -n "$CONDA_ENV" python3)
fi
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
MODEL_BASE=${MODEL_BASE:-""}
REVA_ROOT=${REVA_ROOT:-"$PROJECT_ROOT/data/reva_test"}
REVA_JSON=${REVA_JSON:-"$REVA_ROOT/test_set.json"}
NUM_CHUNKS=${NUM_CHUNKS:-1}
MAX_FRAMES=${MAX_FRAMES:-32}
MAX_PIXELS=${MAX_PIXELS:-$((224*224))}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
BACKEND=${BACKEND:-transformers}
RESUME=${RESUME:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs"}
EVAL_NAME=${EVAL_NAME:-"reva_v2_test"}
GT_FILE=${GT_FILE:-"data/rsvidqa/reva_v2_test_set.json"}

echo "=== ReVA_V2 test Evaluation ==="
echo "Model:      $MODEL_PATH"
echo "ReVA root:  $REVA_ROOT"
echo "ReVA json:  $REVA_JSON"
echo "GPU chunks: $NUM_CHUNKS"
echo "Max frames: $MAX_FRAMES"
echo "Backend:    $BACKEND"
echo "Resume:     $RESUME"

echo ""
echo "--- Step 1: Preparing test set ---"
REVA_ROOT="$REVA_ROOT" REVA_JSON="$REVA_JSON" "${PYTHON[@]}" data/rsvidqa/prepare_reva_v2_test_set.py

echo ""
echo "--- Step 2: Running inference ---"

export MY_PROMPT_TEMPLATE="This is a video with duration {duration} seconds.
Please carefully watch the video and answer the multiple-choice question below.
Think step-by-step within <think> </think> tags, then provide only the letter of the correct option within <answer> </answer> tags.
Question: {input_text}"

OUT_CHUNK_DIR="$OUTPUT_DIR/$EVAL_NAME/$EVAL_NAME"
mkdir -p "$OUT_CHUNK_DIR"

# Reusing predictions from another model or evaluation configuration silently
# invalidates model comparisons. Fresh runs therefore reject existing prediction
# files. RESUME=1 is accepted only when the saved run configuration matches.
shopt -s nullglob
prediction_files=()
for prediction_file in "$OUT_CHUNK_DIR"/*.json; do
    if [ "$(basename "$prediction_file")" != "result.json" ]; then
        prediction_files+=("$prediction_file")
    fi
done
shopt -u nullglob

if [ "${#prediction_files[@]}" -gt 0 ] && [ "$RESUME" != "1" ]; then
    echo "Existing predictions found in: $OUT_CHUNK_DIR" >&2
    echo "Use a new EVAL_NAME for a fresh run, or set RESUME=1 to resume the exact same configuration." >&2
    exit 2
fi

if [ "$RESUME" = "1" ]; then
    for prediction_file in "${prediction_files[@]}"; do
        prediction_name="$(basename "$prediction_file")"
        if [ "$NUM_CHUNKS" -eq 1 ]; then
            if [ "$prediction_name" != "pred.json" ]; then
                echo "Unexpected prediction shard for NUM_CHUNKS=1: $prediction_name" >&2
                exit 2
            fi
        elif [[ ! "$prediction_name" =~ ^${NUM_CHUNKS}_([0-9]+)\.json$ ]] || [ "${BASH_REMATCH[1]}" -ge "$NUM_CHUNKS" ]; then
            echo "Unexpected prediction shard for NUM_CHUNKS=$NUM_CHUNKS: $prediction_name" >&2
            exit 2
        fi
    done
fi

run_config_text="$(printf '%s\n' \
    "MODEL_PATH=$MODEL_PATH" \
    "MODEL_BASE=$MODEL_BASE" \
    "REVA_ROOT=$REVA_ROOT" \
    "REVA_JSON=$REVA_JSON" \
    "NUM_CHUNKS=$NUM_CHUNKS" \
    "MAX_FRAMES=$MAX_FRAMES" \
    "MAX_PIXELS=$MAX_PIXELS" \
    "BACKEND=$BACKEND")"
run_config_file="$OUT_CHUNK_DIR/run_config.txt"

if [ "$RESUME" = "1" ] && [ "${#prediction_files[@]}" -gt 0 ]; then
    if [ ! -f "$run_config_file" ]; then
        echo "Cannot safely resume: missing $run_config_file" >&2
        exit 2
    fi
    if [ "$(cat "$run_config_file")" != "$run_config_text" ]; then
        echo "Cannot safely resume: current configuration differs from $run_config_file" >&2
        exit 2
    fi
else
    printf '%s\n' "$run_config_text" > "$run_config_file"
fi

pids=()
for idx in $(seq 0 $((NUM_CHUNKS - 1))); do
    extra_args=()
    if [ -n "$MODEL_BASE" ]; then
        extra_args+=(--model-base "$MODEL_BASE")
    fi
    CUDA_VISIBLE_DEVICES=$idx "${PYTHON[@]}" inference_vllm_origin_number.py \
        --model-path "$MODEL_PATH" \
        "${extra_args[@]}" \
        --video_dir "$REVA_ROOT" \
        --gt_file "$GT_FILE" \
        --output_dir "$OUT_CHUNK_DIR" \
        --output_name "pred" \
        --num-chunks "$NUM_CHUNKS" \
        --chunk-idx "$idx" \
        --max_pixels "$MAX_PIXELS" \
        --max_frames "$MAX_FRAMES" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --max_model_len "$MAX_MODEL_LEN" \
        --backend "$BACKEND" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid" || { echo "Inference process $pid failed"; exit 1; }
done

echo ""
echo "--- Step 3: Computing accuracy ---"
"${PYTHON[@]}" "$PROJECT_ROOT/scripts/score_reva_predictions.py" \
    --output-dir "$OUT_CHUNK_DIR" \
    --gt-file "$GT_FILE"

echo ""
echo "=== Done ==="
