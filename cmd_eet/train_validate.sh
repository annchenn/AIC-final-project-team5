#!/usr/bin/env bash
set -euo pipefail

export HF_USER=${HF_USER:-ann0000000}
export DATASET_NAME=${DATASET_NAME:-aic-finalproject-dataset-v2}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICE:-}" ]]; then
  echo "[warn] CUDA_VISIBLE_DEVICE is ignored by CUDA; using it as CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICE}" >&2
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICE}
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TMPDIR=${TMPDIR:-${HOME}/tmp}
export WANDB_DIR=${WANDB_DIR:-${HOME}/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${HOME}/wandb}
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export DATASET_ROOT=${DATASET_ROOT:-${SCRIPT_DIR}/datasets/lerobot_cache/${HF_USER}/${DATASET_NAME}}
export RUN_PREFIX=${RUN_PREFIX:-eet-bc}
export POLICIES=${POLICIES:-"diffusion act pi0_fast"}
export SPLIT_MODE=${SPLIT_MODE:-random}
export K_FOLDS=${K_FOLDS:-5}
export FOLDS=${FOLDS:-"0"}
export VAL_FRACTION=${VAL_FRACTION:-0.2}
export SEED=${SEED:-42}
export STEPS=${STEPS:-50000}
export BATCH_SIZE=${BATCH_SIZE:-8}
export NUM_WORKERS=${NUM_WORKERS:-4}
export WANDB_ENABLE=${WANDB_ENABLE:-true}
export IMAGE_TRANSFORMS=${IMAGE_TRANSFORMS:-true}
export VALIDATE_TRAIN_LOSS=${VALIDATE_TRAIN_LOSS:-true}
export VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-8}
export VALIDATION_MAX_BATCHES=${VALIDATION_MAX_BATCHES:-0}

STAMP=$(date +%m%d-%H%M)
RUN_GROUP=${RUN_GROUP:-${RUN_PREFIX}-${SPLIT_MODE}-${STAMP}}
OUTPUT_BASE=${OUTPUT_BASE:-${SCRIPT_DIR}/checkpoints/${RUN_GROUP}}
SPLIT_DIR=${SPLIT_DIR:-${OUTPUT_BASE}/splits}
RESULTS_JSONL=${RESULTS_JSONL:-${OUTPUT_BASE}/validation_results.jsonl}
mkdir -p "$OUTPUT_BASE" "$SPLIT_DIR"
python "${SCRIPT_DIR}/cmd_eet/patch_lerobot_sampler.py"

for POLICY in $POLICIES; do
  for FOLD in $FOLDS; do
    SPLIT_FILE=${SPLIT_DIR}/${POLICY}_fold${FOLD}_${SPLIT_MODE}.json
    eval "$(
      python "${SCRIPT_DIR}/cmd_eet/split_episodes.py" \
        --dataset-root "$DATASET_ROOT" \
        --output "$SPLIT_FILE" \
        --mode "$SPLIT_MODE" \
        --k-folds "$K_FOLDS" \
        --fold "$FOLD" \
        --val-fraction "$VAL_FRACTION" \
        --seed "$SEED" \
        --shell
    )"

    RUN_NAME=${RUN_GROUP}-${POLICY}-fold${FOLD}
    CV_OUTPUT_DIR=${OUTPUT_BASE}/${POLICY}/fold${FOLD}
    LOG_DIR=${OUTPUT_BASE}/logs/${POLICY}/fold${FOLD}
    mkdir -p "$LOG_DIR"

    EXTRA_ARGS=()
    if [[ "$POLICY" == "pi0_fast" ]]; then
      EXTRA_ARGS+=(--policy.use_amp=true --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --batch_size=1)
    else
      EXTRA_ARGS+=(--batch_size="$BATCH_SIZE")
    fi
    if [[ "$IMAGE_TRANSFORMS" == "true" ]]; then
      EXTRA_ARGS+=(--dataset.image_transforms.enable=true)
    fi

    {
      printf '$ lerobot-train'
      printf ' %q' \
        --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
        --dataset.root="$DATASET_ROOT" \
        "--dataset.episodes=${CV_TRAIN_EPISODES}" \
        --policy.type="$POLICY" \
        --output_dir="$CV_OUTPUT_DIR" \
        --job_name="$RUN_NAME" \
        --policy.device=cuda \
        --wandb.enable="$WANDB_ENABLE" \
        --policy.push_to_hub=false \
        --steps="$STEPS" \
        --num_workers="$NUM_WORKERS" \
        --policy.repo_id="${HF_USER}/${RUN_NAME}" \
        "${EXTRA_ARGS[@]}"
      printf '\n'
    } | tee "${LOG_DIR}/train_command.log"

    lerobot-train \
      --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
      --dataset.root="$DATASET_ROOT" \
      "--dataset.episodes=${CV_TRAIN_EPISODES}" \
      --policy.type="$POLICY" \
      --output_dir="$CV_OUTPUT_DIR" \
      --job_name="$RUN_NAME" \
      --policy.device=cuda \
      --wandb.enable="$WANDB_ENABLE" \
      --policy.push_to_hub=false \
      --steps="$STEPS" \
      --num_workers="$NUM_WORKERS" \
      --policy.repo_id="${HF_USER}/${RUN_NAME}" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "${LOG_DIR}/train.log"

    CHECKPOINT_PATH=${CV_OUTPUT_DIR}/checkpoints/last/pretrained_model
    if [[ ! -d "$CHECKPOINT_PATH" ]]; then
      CHECKPOINT_PATH=${CV_OUTPUT_DIR}/pretrained_model
    fi

    python "${SCRIPT_DIR}/cmd_eet/validate_lerobot_loss.py" \
      --policy-checkpoint "$CHECKPOINT_PATH" \
      --dataset-repo-id "${HF_USER}/${DATASET_NAME}" \
      --dataset-root "$DATASET_ROOT" \
      --episodes "$CV_VAL_EPISODES" \
      --batch-size "$VALIDATION_BATCH_SIZE" \
      --num-workers 0 \
      --device cuda \
      --max-batches "$VALIDATION_MAX_BATCHES" \
      --output-json "$RESULTS_JSONL" \
      2>&1 | tee "${LOG_DIR}/validation.log"

    if [[ "$VALIDATE_TRAIN_LOSS" == "true" ]]; then
      python "${SCRIPT_DIR}/cmd_eet/validate_lerobot_loss.py" \
        --policy-checkpoint "$CHECKPOINT_PATH" \
        --dataset-repo-id "${HF_USER}/${DATASET_NAME}" \
        --dataset-root "$DATASET_ROOT" \
        --episodes "$CV_TRAIN_EPISODES" \
        --batch-size "$VALIDATION_BATCH_SIZE" \
        --num-workers 0 \
        --device cuda \
        --max-batches "$VALIDATION_MAX_BATCHES" \
        --output-json "$RESULTS_JSONL" \
        2>&1 | tee "${LOG_DIR}/train_loss_eval.log"
    fi
  done
done

python "${SCRIPT_DIR}/cmd_eet/summarize_validation.py" "$RESULTS_JSONL" | tee "${OUTPUT_BASE}/summary.txt"
