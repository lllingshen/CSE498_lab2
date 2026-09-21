# CSE498 Lab 2

ReVA video question answering with Qwen3-VL-4B-Instruct, Qwen LoRA fine-tuning, and VILA1.5-3b.

Based on the course [starter repository](https://github.com/likaiw2/VLM_evaluating_finetuing). The completed code uses 200 training questions and all 4,000 official test questions, with at most four frames per video.

| Model | Correct / total | Accuracy |
|---|---:|---:|
| Qwen base | 2,675 / 4,000 | 66.88% |
| Qwen LoRA | 2,673 / 4,000 | 66.83% |
| VILA base | 1,940 / 4,000 | 48.50% |

Scores are in [outputs/model_comparison.csv](outputs/model_comparison.csv). [RESULTS.md](RESULTS.md) describes the training settings and three examples. Videos, weights, environments, and raw prediction logs are downloaded or generated locally.

## 1. Set up Qwen and download data

The experiments ran with Python 3.10 and PyTorch 2.7.1 on an RTX 5090. Qwen and VILA need separate environments because they use different Transformers versions. Run commands from this repository's root.

```bash
python3.10 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-qwen.txt
unset PYTHONPATH
export PYTHONNOUSERSITE=1 CONDA_ENV=
export MODEL_PATH="$PWD/models/Qwen3-VL-4B-Instruct"
export REVA_ROOT="$PWD/data/ReVA"
export REVA_JSON="$PWD/data/reva_test/test_set.json"
export BACKEND=transformers MAX_FRAMES=4
export FORCE_QWENVL_VIDEO_READER=torchvision

hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$MODEL_PATH"
python scripts/download_reva.py
bash scripts/prepare_qwen_train_data.sh
```

The download script retrieves the [official ReVA annotations](https://huggingface.co/datasets/ReVA-Benchmark/ReVA), the referenced videos, and creates the training video links.

## 2. Evaluate Qwen before fine-tuning

```bash
bash scripts/run_eval_qwen_base.sh
```

Score: `outputs/qwen_base/qwen_base/result.csv`.

## 3. Fine-tune Qwen with LoRA

```bash
NPROC_PER_NODE=1 BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 EPOCHS=1 \
SAVE_STEPS=50 MAX_PIXELS=50176 VIDEO_MAX_FRAMES=4 VIDEO_FPS=1 \
MODEL_MAX_LENGTH=2048 USE_DEEPSPEED=0 DATA_FLATTEN=False \
ATTN_IMPLEMENTATION=sdpa DATALOADER_NUM_WORKERS=2 \
bash scripts/run_finetune_qwen.sh
```

This uses the supplied learning rate of `2e-7`, LoRA rank 8, alpha 16, and dropout 0. The 200 questions produce 50 optimizer steps. The adapter is saved to `outputs/qwen_reva_sft/`.

## 4. Evaluate Qwen after fine-tuning

```bash
MODEL_BASE="$PWD/models/Qwen3-VL-4B-Instruct" \
MODEL_PATH="$PWD/outputs/qwen_reva_sft" \
bash scripts/run_eval_qwen_finetuned.sh
```

Score: `outputs/qwen_sft/qwen_sft/result.csv`.

For a fresh repeat, use a new `EVAL_NAME`; to continue an interrupted evaluation with the same settings, use `RESUME=1`. For a new training run, set a new absolute `OUTPUT_DIR`.

## 5. Evaluate VILA

```bash
deactivate
python3.10 -m venv .venv-vila
source .venv-vila/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vila.txt
mkdir -p work
git clone https://github.com/NVlabs/VILA.git work/VILA
git -C work/VILA checkout 0f1426e8da9181e6e6653e10bc15f62d515fa2f6
git -C work/VILA apply "$PWD/patches/vila-compat.patch"
hf download Efficient-Large-Model/VILA1.5-3b --local-dir models/VILA1.5-3b

PYTHONPATH="$PWD/work/VILA" python vila_eval/reva_v2.py \
  --model-path "$PWD/models/VILA1.5-3b" \
  --question-file "$PWD/data/reva_test/test_set.json" \
  --dataset-root "$PWD/data/ReVA" \
  --output-dir "$PWD/outputs/vila_reva_v2" \
  --gpu 0 --num-video-frames 4 --video-max-tiles 1 \
  --generation-config '{"do_sample": false, "max_new_tokens": 128}'
```

The small patch enables the SDPA inference path used in this experiment. VILA's original license is included in `patches/VILA-LICENSE`. VILA is evaluated without fine-tuning.

Score: `outputs/vila_reva_v2/metrics.json`. Add `--resume` to continue an interrupted VILA run.

## 6. Compare results

```bash
python scripts/compare_model_metrics.py
cat outputs/model_comparison.csv
```

Missing or unparseable answers count as wrong. The repository contains the saved comparison and all three score files. The seven tests supplied with the assignment are retained in `tests/`.

The main completed files are `scripts/build_qwen_train_data.py`, `reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py`, `scripts/score_reva_predictions.py`, and `vila_eval/reva_v2.py`. Qwen training and evaluation also include the local compatibility changes needed for this run. Original source notices are retained.
