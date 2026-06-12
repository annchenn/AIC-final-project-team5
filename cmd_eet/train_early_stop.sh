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
export RUN_PREFIX=${RUN_PREFIX:-eet-bc-early}
export POLICIES=${POLICIES:-"diffusion"}
export SPLIT_MODE=${SPLIT_MODE:-random}
export K_FOLDS=${K_FOLDS:-5}
export FOLDS=${FOLDS:-"0"}
export VAL_FRACTION=${VAL_FRACTION:-0.2}
export SEED=${SEED:-42}
export MAX_STEPS=${MAX_STEPS:-100000}
export CHUNK_STEPS=${CHUNK_STEPS:-10000}
export PATIENCE=${PATIENCE:-3}
export MIN_DELTA=${MIN_DELTA:-0.0}
export BATCH_SIZE=${BATCH_SIZE:-8}
export NUM_WORKERS=${NUM_WORKERS:-4}
export VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-8}
export VALIDATION_MAX_BATCHES=${VALIDATION_MAX_BATCHES:-0}
export WANDB_ENABLE=${WANDB_ENABLE:-true}
export IMAGE_TRANSFORMS=${IMAGE_TRANSFORMS:-true}
export VALIDATE_TRAIN_LOSS=${VALIDATE_TRAIN_LOSS:-false}
export ACT_CHUNK_SIZE=${ACT_CHUNK_SIZE:-}
export ACT_N_ACTION_STEPS=${ACT_N_ACTION_STEPS:-}

STAMP=$(date +%m%d-%H%M)
RUN_GROUP=${RUN_GROUP:-${RUN_PREFIX}-${SPLIT_MODE}-${STAMP}}
OUTPUT_BASE=${OUTPUT_BASE:-${SCRIPT_DIR}/checkpoints/${RUN_GROUP}}
SPLIT_DIR=${SPLIT_DIR:-${OUTPUT_BASE}/splits}
LOG_BASE=${LOG_BASE:-${OUTPUT_BASE}/logs}
RESULTS_JSONL=${RESULTS_JSONL:-${OUTPUT_BASE}/early_stop_results.jsonl}
mkdir -p "$OUTPUT_BASE" "$SPLIT_DIR" "$LOG_BASE"
python "${SCRIPT_DIR}/cmd_eet/patch_lerobot_sampler.py"

if (( CHUNK_STEPS <= 0 || MAX_STEPS <= 0 )); then
  echo "CHUNK_STEPS and MAX_STEPS must be positive." >&2
  exit 2
fi

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
    LOG_DIR=${LOG_BASE}/${POLICY}/fold${FOLD}
    mkdir -p "$LOG_DIR"

    EXTRA_ARGS=()
    if [[ "$POLICY" == "pi0_fast" ]]; then
      EXTRA_ARGS+=(--policy.use_amp=true --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --batch_size=1)
    else
      EXTRA_ARGS+=(--batch_size="$BATCH_SIZE")
    fi
    if [[ "$POLICY" == "act" ]]; then
      if [[ -n "$ACT_CHUNK_SIZE" ]]; then
        EXTRA_ARGS+=(--policy.chunk_size="$ACT_CHUNK_SIZE")
      fi
      if [[ -n "$ACT_N_ACTION_STEPS" ]]; then
        EXTRA_ARGS+=(--policy.n_action_steps="$ACT_N_ACTION_STEPS")
      fi
    fi
    if [[ "$IMAGE_TRANSFORMS" == "true" ]]; then
      EXTRA_ARGS+=(--dataset.image_transforms.enable=true)
    fi

    BEST_LOSS="inf"
    BEST_STEP=0
    BEST_CHECKPOINT=""
    BAD_ROUNDS=0
    CURRENT_STEPS=0
    RESUME=false

    LAST_CONFIG=${CV_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
    if [[ -f "$LAST_CONFIG" ]]; then
      CURRENT_STEPS=$(python - "$CV_OUTPUT_DIR" <<'PY_INNER'
from pathlib import Path
import sys
ckpt_root = Path(sys.argv[1]) / "checkpoints"
steps = []
for path in ckpt_root.iterdir() if ckpt_root.exists() else []:
    if path.is_dir() and path.name.isdigit():
        steps.append(int(path.name))
print(max(steps) if steps else 0)
PY_INNER
)
      RESUME=true
      if [[ -f "${LOG_DIR}/best_checkpoint.txt" ]]; then
        BEST_STEP=$(awk -F= '$1 == "best_step" {print $2}' "${LOG_DIR}/best_checkpoint.txt")
        BEST_LOSS=$(awk -F= '$1 == "best_loss" {print $2}' "${LOG_DIR}/best_checkpoint.txt")
        BEST_CHECKPOINT=$(awk -F= '$1 == "best_checkpoint" {print $2}' "${LOG_DIR}/best_checkpoint.txt")
      fi
      echo "Resuming ${RUN_NAME} from step ${CURRENT_STEPS}; best_step=${BEST_STEP}, best_loss=${BEST_LOSS}." | tee -a "${LOG_DIR}/early_stop.log"
    fi

    while (( CURRENT_STEPS < MAX_STEPS )); do
      TARGET_STEPS=$((CURRENT_STEPS + CHUNK_STEPS))
      if (( TARGET_STEPS > MAX_STEPS )); then
        TARGET_STEPS=$MAX_STEPS
      fi

      echo "=== ${RUN_NAME}: train until step ${TARGET_STEPS} ===" | tee -a "${LOG_DIR}/early_stop.log"

      if [[ "$RESUME" == "false" ]]; then
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
          --steps="$TARGET_STEPS" \
          --save_freq="$TARGET_STEPS" \
          --num_workers="$NUM_WORKERS" \
          --policy.repo_id="${HF_USER}/${RUN_NAME}" \
          "${EXTRA_ARGS[@]}" \
          2>&1 | tee "${LOG_DIR}/train_step${TARGET_STEPS}.log"
      else
        CONFIG_PATH=${CV_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
        lerobot-train \
          --config_path="$CONFIG_PATH" \
          --resume=true \
          --steps="$TARGET_STEPS" \
          --save_freq="$CHUNK_STEPS" \
          --wandb.enable="$WANDB_ENABLE" \
          2>&1 | tee "${LOG_DIR}/train_step${TARGET_STEPS}.log"
      fi

      CHECKPOINT_STEP_NAME=$(printf "%06d" "$TARGET_STEPS")
      CHECKPOINT_PATH=${CV_OUTPUT_DIR}/checkpoints/${CHECKPOINT_STEP_NAME}/pretrained_model
      if [[ ! -d "$CHECKPOINT_PATH" ]]; then
        CHECKPOINT_PATH=${CV_OUTPUT_DIR}/checkpoints/last/pretrained_model
      fi
      STEP_RESULT=${LOG_DIR}/validation_step${TARGET_STEPS}.jsonl
      python "${SCRIPT_DIR}/cmd_eet/validate_lerobot_loss.py" \
        --policy-checkpoint "$CHECKPOINT_PATH" \
        --dataset-repo-id "${HF_USER}/${DATASET_NAME}" \
        --dataset-root "$DATASET_ROOT" \
        --episodes "$CV_VAL_EPISODES" \
        --batch-size "$VALIDATION_BATCH_SIZE" \
        --num-workers 0 \
        --device cuda \
        --max-batches "$VALIDATION_MAX_BATCHES" \
        --step "$TARGET_STEPS" \
        --split validation \
        --output-json "$STEP_RESULT" \
        2>&1 | tee "${LOG_DIR}/validation_step${TARGET_STEPS}.log"
      cat "$STEP_RESULT" >> "$RESULTS_JSONL"

      VAL_LOSS=$(python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["loss"])' "$STEP_RESULT")
      IMPROVED=$(python -c 'import math,sys; val=float(sys.argv[1]); best=float(sys.argv[2]) if sys.argv[2] != "inf" else math.inf; delta=float(sys.argv[3]); print("1" if val < best - delta else "0")' "$VAL_LOSS" "$BEST_LOSS" "$MIN_DELTA")

      if [[ "$IMPROVED" == "1" ]]; then
        BEST_LOSS=$VAL_LOSS
        BEST_STEP=$TARGET_STEPS
        BEST_CHECKPOINT=$CHECKPOINT_PATH
        BAD_ROUNDS=0
        echo "best_step=${BEST_STEP}" > "${LOG_DIR}/best_checkpoint.txt"
        echo "best_loss=${BEST_LOSS}" >> "${LOG_DIR}/best_checkpoint.txt"
        echo "best_checkpoint=${BEST_CHECKPOINT}" >> "${LOG_DIR}/best_checkpoint.txt"
        echo "New best validation loss: ${BEST_LOSS} at step ${BEST_STEP}" | tee -a "${LOG_DIR}/early_stop.log"
      else
        BAD_ROUNDS=$((BAD_ROUNDS + 1))
        echo "Validation did not improve: loss=${VAL_LOSS}, best=${BEST_LOSS}, bad_rounds=${BAD_ROUNDS}/${PATIENCE}" | tee -a "${LOG_DIR}/early_stop.log"
      fi

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
          --step "$TARGET_STEPS" \
          --split train \
          --output-json "$RESULTS_JSONL" \
          2>&1 | tee "${LOG_DIR}/train_loss_step${TARGET_STEPS}.log"
      fi

      CURRENT_STEPS=$TARGET_STEPS
      RESUME=true
      if (( BAD_ROUNDS >= PATIENCE )); then
        echo "Early stopping at step ${CURRENT_STEPS}; best step ${BEST_STEP}, best validation loss ${BEST_LOSS}." | tee -a "${LOG_DIR}/early_stop.log"
        break
      fi
    done
  done
done

python "${SCRIPT_DIR}/cmd_eet/summarize_validation.py" "$RESULTS_JSONL" | tee "${OUTPUT_BASE}/summary.txt"
