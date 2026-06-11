#!/usr/bin/env bash
set -euo pipefail

export DATASET_OWNER=${DATASET_OWNER:-yun0523}
export HF_USER=${HF_USER:-mikehsuhoodie}
export DATASET_NAME=${DATASET_NAME:-moving-cube-grasp}
export RUN_NAME=${1:-moving_cube_smolvla_temporal_early_stop}

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SOURCE_DATASET_ROOT=${SOURCE_DATASET_ROOT:-${SCRIPT_DIR}/datasets/lerobot_cache/${DATASET_OWNER}/${DATASET_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/${RUN_NAME}}
EARLY_STOP_DIR=${EARLY_STOP_DIR:-${SCRIPT_DIR}/checkpoints/${RUN_NAME}_early_stop}
SPLIT_PATH=${SPLIT_PATH:-${EARLY_STOP_DIR}/episode_split.json}
VAL_LOG=${VAL_LOG:-${EARLY_STOP_DIR}/validation_history.jsonl}
EARLY_STOP_STATE=${EARLY_STOP_STATE:-${EARLY_STOP_DIR}/early_stop_state.json}

MAX_STEPS=${MAX_STEPS:-200000}
EVAL_EVERY=${EVAL_EVERY:-20000}
PATIENCE=${PATIENCE:-3}
MIN_DELTA=${MIN_DELTA:-1e-4}
VAL_RATIO=${VAL_RATIO:-0.1}
SPLIT_SEED=${SPLIT_SEED:-42}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-0}
VAL_MAX_BATCHES=${VAL_MAX_BATCHES:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
N_ACTION_STEPS=${N_ACTION_STEPS:-4}
WANDB_ENABLE=${WANDB_ENABLE:-true}

mkdir -p "${EARLY_STOP_DIR}" ~/tmp ~/wandb
export TMPDIR=${TMPDIR:-~/tmp}
export WANDB_DIR=${WANDB_DIR:-~/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-~/wandb}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export VLM_MODEL_NAME=${VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}

if [[ ! -f "${SPLIT_PATH}" ]]; then
  uv run python scripts/split_lerobot_episodes.py \
    --dataset-repo-id "${DATASET_OWNER}/${DATASET_NAME}" \
    --dataset-root "${SOURCE_DATASET_ROOT}" \
    --output "${SPLIT_PATH}" \
    --val-ratio "${VAL_RATIO}" \
    --seed "${SPLIT_SEED}"
fi

export TRAIN_EPISODES_JSON="${SPLIT_PATH}"

best_loss="inf"
best_checkpoint=""
patience_counter=0
last_checkpoint=""
current_step=0

if [[ -L "${OUTPUT_DIR}/checkpoints/last" ]]; then
  last_checkpoint=$(uv run python -c "from pathlib import Path; print((Path('${OUTPUT_DIR}') / 'checkpoints' / 'last').resolve())")
  current_step=$(uv run python -c "import json; print(json.load(open('${last_checkpoint}/training_state/training_step.json'))['step'])")
  if [[ -f "${EARLY_STOP_STATE}" ]]; then
    best_loss=$(uv run python -c "import json; data=json.load(open('${EARLY_STOP_STATE}')); print(data.get('best_loss', 'inf'))")
    best_checkpoint=$(uv run python -c "import json; data=json.load(open('${EARLY_STOP_STATE}')); print(data.get('best_checkpoint', ''))")
    patience_counter=$(uv run python -c "import json; data=json.load(open('${EARLY_STOP_STATE}')); print(data.get('patience_counter', 0))")
  fi
  echo "[early-stop] resuming orchestration from step ${current_step}, checkpoint ${last_checkpoint}"
else
  : > "${VAL_LOG}"
fi

if [[ -d "${OUTPUT_DIR}" && -z "${last_checkpoint}" && ! -d "${OUTPUT_DIR}/checkpoints" ]]; then
  echo "[early-stop] ERROR: OUTPUT_DIR already exists before first training segment: ${OUTPUT_DIR}" >&2
  echo "[early-stop] It looks like a previous failed setup-only run created this directory." >&2
  echo "[early-stop] Use a new RUN_NAME or move/delete that directory if it has no training checkpoints." >&2
  exit 1
fi

while (( current_step < MAX_STEPS )); do
  next_step=$(( current_step + EVAL_EVERY ))
  if (( next_step > MAX_STEPS )); then
    next_step=${MAX_STEPS}
  fi

  train_args=(
    --dataset.repo_id="${DATASET_OWNER}/${DATASET_NAME}"
    --dataset.root="${SOURCE_DATASET_ROOT}"
    --dataset.video_backend=pyav
    --policy.type=smolvla
    --policy.chunk_size=16
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.vlm_model_name="${VLM_MODEL_NAME}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${RUN_NAME}"
    --policy.device=cuda
    --policy.push_to_hub=false
    --wandb.enable="${WANDB_ENABLE}"
    --policy.repo_id="${HF_USER}/aic-finalproject-team5-smolvla-temporal"
    --steps="${next_step}"
    --save_checkpoint=true
    --save_freq="${EVAL_EVERY}"
    --eval_freq=0
  )

  if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
    train_args+=(--batch_size="${TRAIN_BATCH_SIZE}")
  fi

  if [[ -n "${last_checkpoint}" ]]; then
    train_args+=(--resume=true --config_path="${last_checkpoint}/pretrained_model/train_config.json")
  fi

  echo "[early-stop] training until step ${next_step}"
  uv run python scripts/train_smolvla_temporal_debug.py "${train_args[@]}"

  checkpoint_dir=$(uv run python -c "from pathlib import Path; step=${next_step}; total=${next_step}; print(Path('${OUTPUT_DIR}') / 'checkpoints' / f'{step:0{max(6, len(str(total)))}d}')")
  last_checkpoint="${checkpoint_dir}"

  val_json="${EARLY_STOP_DIR}/validation_step_${next_step}.json"
  uv run python scripts/validate_lerobot_checkpoint.py \
    --checkpoint "${checkpoint_dir}" \
    --split "${SPLIT_PATH}" \
    --dataset-repo-id "${DATASET_OWNER}/${DATASET_NAME}" \
    --dataset-root "${SOURCE_DATASET_ROOT}" \
    --batch-size "${VAL_BATCH_SIZE}" \
    --num-workers "${VAL_NUM_WORKERS}" \
    --device cuda \
    --video-backend pyav \
    --max-batches "${VAL_MAX_BATCHES}" \
    --output "${val_json}"

  val_loss=$(uv run python -c "import json; print(json.load(open('${val_json}'))['validation_loss'])")

  decision=$(uv run python -c "import json, math; best=float('${best_loss}') if '${best_loss}' != 'inf' else math.inf; loss=float('${val_loss}'); pc=int('${patience_counter}'); improved=loss < best - float('${MIN_DELTA}'); best=loss if improved else best; pc=0 if improved else pc+1; print(json.dumps({'best_loss': best, 'patience_counter': pc, 'improved': improved}))")
  best_loss=$(DECISION="${decision}" uv run python -c "import json, os; print(json.loads(os.environ['DECISION'])['best_loss'])")
  patience_counter=$(DECISION="${decision}" uv run python -c "import json, os; print(json.loads(os.environ['DECISION'])['patience_counter'])")
  improved=$(DECISION="${decision}" uv run python -c "import json, os; print(str(json.loads(os.environ['DECISION'])['improved']).lower())")

  if [[ "${improved}" == "true" ]]; then
    best_checkpoint="${checkpoint_dir}"
  fi

  STEP="${next_step}" CHECKPOINT_DIR="${checkpoint_dir}" VAL_LOSS="${val_loss}" BEST_CHECKPOINT="${best_checkpoint}" BEST_LOSS="${best_loss}" PATIENCE_COUNTER="${patience_counter}" PATIENCE_VALUE="${PATIENCE}" MIN_DELTA_VALUE="${MIN_DELTA}" uv run python -c "import json, os; print(json.dumps({'step': int(os.environ['STEP']), 'checkpoint_path': os.environ['CHECKPOINT_DIR'], 'validation_loss': float(os.environ['VAL_LOSS']), 'best_checkpoint': os.environ['BEST_CHECKPOINT'], 'best_loss': float(os.environ['BEST_LOSS']), 'patience_counter': int(os.environ['PATIENCE_COUNTER']), 'patience': int(os.environ['PATIENCE_VALUE']), 'min_delta': float(os.environ['MIN_DELTA_VALUE'])}))" >> "${VAL_LOG}"

  STEP="${next_step}" CHECKPOINT_DIR="${checkpoint_dir}" VAL_LOSS="${val_loss}" BEST_CHECKPOINT="${best_checkpoint}" BEST_LOSS="${best_loss}" PATIENCE_COUNTER="${patience_counter}" PATIENCE_VALUE="${PATIENCE}" uv run python -c "import json, os; state={'step': int(os.environ['STEP']), 'checkpoint_path': os.environ['CHECKPOINT_DIR'], 'validation_loss': float(os.environ['VAL_LOSS']), 'best_checkpoint': os.environ['BEST_CHECKPOINT'], 'best_loss': float(os.environ['BEST_LOSS']), 'patience_counter': int(os.environ['PATIENCE_COUNTER']), 'patience': int(os.environ['PATIENCE_VALUE']), 'stopped': int(os.environ['PATIENCE_COUNTER']) >= int(os.environ['PATIENCE_VALUE'])}; print(json.dumps(state, indent=2))" > "${EARLY_STOP_STATE}"

  echo "[early-stop] step=${next_step} val_loss=${val_loss} best=${best_loss} best_checkpoint=${best_checkpoint} patience=${patience_counter}/${PATIENCE}"

  if (( patience_counter >= PATIENCE )); then
    echo "[early-stop] stopping: validation loss did not improve for ${PATIENCE} validations"
    break
  fi

  current_step=${next_step}
done

echo "[early-stop] best checkpoint: ${best_checkpoint}"
echo "[early-stop] validation log: ${VAL_LOG}"
