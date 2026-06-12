#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export POLICY=${POLICY:-pi0_fast}
export RUN_GROUP=${RUN_GROUP:-eet-bc-random-latest}
export FOLD=${FOLD:-0}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/${RUN_GROUP}/${POLICY}/fold${FOLD}/checkpoints/last/pretrained_model}
export EVAL_TASK=${EVAL_TASK:-eval/toy_blocks_collection_eval.py}
export ACTION_HORIZON=${ACTION_HORIZON:-8}
export EVAL_ROUNDS=${EVAL_ROUNDS:-30}
export EPISODE_LENGTH_S=${EPISODE_LENGTH_S:-60}
export OBJECT_POSES=${OBJECT_POSES:-${SCRIPT_DIR}/data/object_poses_clear_aug.json}
export PROGRESS_INTERVAL_S=${PROGRESS_INTERVAL_S:-10}
export FIRST_PICK_OBJECT=${FIRST_PICK_OBJECT:-}
export FIRST_PICK_LIFT_Z=${FIRST_PICK_LIFT_Z:-0.12}
export DEBUG_GRASP_METRICS=${DEBUG_GRASP_METRICS:-0}

# Inside the IsaacLab Docker container this repo is mounted at /workspace/aicapstone,
# while host-side notes/logs often use /project/.../final-project. Rewrite that
# common host path so local LeRobot loading sees a real directory.
if [[ ! -d "$CHECKPOINT_PATH" && "$CHECKPOINT_PATH" == /project/*/final-project/* ]]; then
  RELATIVE_CHECKPOINT=${CHECKPOINT_PATH#*/final-project/}
  CANDIDATE_CHECKPOINT=${SCRIPT_DIR}/${RELATIVE_CHECKPOINT}
  if [[ -d "$CANDIDATE_CHECKPOINT" ]]; then
    echo "[eval] Rewriting host checkpoint path to container path: $CANDIDATE_CHECKPOINT"
    CHECKPOINT_PATH=$CANDIDATE_CHECKPOINT
  fi
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "[eval] Checkpoint directory not found: $CHECKPOINT_PATH" >&2
  echo "[eval] In Docker, use a path under ${SCRIPT_DIR}/checkpoints/..." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "$FIRST_PICK_OBJECT" ]]; then
  EXTRA_ARGS+=(--first_pick_object="$FIRST_PICK_OBJECT")
  EXTRA_ARGS+=(--first_pick_lift_z="$FIRST_PICK_LIFT_Z")
fi
if [[ "$DEBUG_GRASP_METRICS" == "1" ]]; then
  EXTRA_ARGS+=(--debug_grasp_metrics)
fi

python scripts/rollout_pose_replay.py \
  --task="$EVAL_TASK" \
  --policy_type="lerobot-${POLICY}" \
  --policy_checkpoint_path="$CHECKPOINT_PATH" \
  --policy_action_horizon="$ACTION_HORIZON" \
  --device=cuda \
  --enable_cameras \
  --eval_rounds="$EVAL_ROUNDS" \
  --episode_length_s="$EPISODE_LENGTH_S" \
  --object_poses="$OBJECT_POSES" \
  --progress_interval_s="$PROGRESS_INTERVAL_S" \
  "${EXTRA_ARGS[@]}"
