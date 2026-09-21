#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(python3 -c 'import random; print(random.randint(20001, 29999))')}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json
use_deepspeed=${USE_DEEPSPEED:-1}

# Model configuration
llm=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}

# Training hyperparameters
lr=${LR:-2e-7}
batch_size=${BATCH_SIZE:-1}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
epochs=${EPOCHS:-1}
save_steps=${SAVE_STEPS:-1000}
max_pixels=${MAX_PIXELS:-50176}
model_max_length=${MODEL_MAX_LENGTH:-8192}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4}
video_max_frames=${VIDEO_MAX_FRAMES:-4}
video_min_frames=${VIDEO_MIN_FRAMES:-2}
video_max_pixels=${VIDEO_MAX_PIXELS:-50176}
video_min_pixels=${VIDEO_MIN_PIXELS:-784}
video_fps=${VIDEO_FPS:-1}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration. Defined in qwenvl/data/__init__.py.
datasets=${DATASETS:-"reva_train_small"}

# Output configuration
run_name=${RUN_NAME:-"qwen_reva_sft"}
output_dir=${OUTPUT_DIR:-"../outputs/qwen_reva_sft"}
report_to=${REPORT_TO:-"none"}

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten ${DATA_FLATTEN:-True} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --lora_enable True \
    --lora_r ${LORA_R:-8} \
    --lora_alpha ${LORA_ALPHA:-16} \
    --lora_dropout ${LORA_DROPOUT:-0.0} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels 784 \
    --video_max_frames ${video_max_frames} \
    --video_min_frames ${video_min_frames} \
    --video_max_pixels ${video_max_pixels} \
    --video_min_pixels ${video_min_pixels} \
    --video_fps ${video_fps} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers} \
    --run_name ${run_name} \
    --report_to ${report_to}"

if [ "$use_deepspeed" = "1" ]; then
    args="--deepspeed ${deepspeed} ${args}"
fi

echo "=== Qwen SFT ==="
echo "Model:    ${llm}"
echo "Dataset:  ${datasets}"
echo "Output:   ${output_dir}"
echo "GPUs:     ${NPROC_PER_NODE}"
echo "DeepSpeed:${use_deepspeed}"

# Launch training
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    python3 ${entry_file} ${args}
else
    python3 -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
fi
