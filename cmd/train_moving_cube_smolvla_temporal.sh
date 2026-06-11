#!/usr/bin/env bash
set -euo pipefail

export DATASET_OWNER=${DATASET_OWNER:-yun0523}
export HF_USER=${HF_USER:-mikehsuhoodie}
export DATASET_NAME=${DATASET_NAME:-moving-cube-grasp}
export RUN_NAME=${1:-moving_cube_smolvla_temporal}

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
SOURCE_DATASET_ROOT=${SOURCE_DATASET_ROOT:-${SCRIPT_DIR}/datasets/lerobot_cache/${DATASET_OWNER}/${DATASET_NAME}}

mkdir -p ~/tmp ~/wandb
export TMPDIR=${TMPDIR:-~/tmp}
export WANDB_DIR=${WANDB_DIR:-~/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-~/wandb}
# First SmolVLA run needs Hugging Face access to cache the VLM config/processor.
# After models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct exists locally,
# you can run with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export VLM_MODEL_NAME=${VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}

# Moving cube is a sliding target, so we keep n_action_steps at 4. The policy
# still predicts 16 actions per chunk, but replans frequently enough to react
# when the cube keeps moving during approach and grasp.
TRAIN_STEPS=${TRAIN_STEPS:-}
N_ACTION_STEPS=${N_ACTION_STEPS:-4}
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  TRAIN_STEPS=${TRAIN_STEPS:-10}
  WANDB_ENABLE=${WANDB_ENABLE:-false}
else
  WANDB_ENABLE=${WANDB_ENABLE:-true}
fi

STEP_ARGS=()
if [[ -n "${TRAIN_STEPS}" ]]; then
  STEP_ARGS+=(--steps="${TRAIN_STEPS}" --save_checkpoint=false --eval_freq=0)
fi

uv run python scripts/train_smolvla_temporal_debug.py \
  --dataset.repo_id="${DATASET_OWNER}/${DATASET_NAME}" \
  --dataset.root="${SOURCE_DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=smolvla \
  --policy.chunk_size=16 \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.vlm_model_name="${VLM_MODEL_NAME}" \
  --output_dir="${SCRIPT_DIR}/checkpoints/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable="${WANDB_ENABLE}" \
  --policy.repo_id="${HF_USER}/aic-finalproject-team5-smolvla-temporal" \
  "${STEP_ARGS[@]}"
