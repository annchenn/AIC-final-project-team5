#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export HF_USER=${HF_USER:-ann0000000}
export DATASET_NAME=${DATASET_NAME:-aic-finalproject-dataset-no-basket-fsm}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${SCRIPT_DIR}/datasets/lerobot_cache}
export OBJECT_POSES=${OBJECT_POSES:-${SCRIPT_DIR}/data/object_poses_clear_aug.json}
export RUNS=${RUNS:-1}
export RESUME=${RESUME:-0}
export TOY_FSM_DEBUG_INTERVAL=${TOY_FSM_DEBUG_INTERVAL:-0}
export TOY_GRASP_XY_GREEN=${TOY_GRASP_XY_GREEN:-}
export TOY_GRASP_XY_BLUE=${TOY_GRASP_XY_BLUE:-}
export TOY_GRASP_XY_RED=${TOY_GRASP_XY_RED:-}
export TOY_GRASP_RETREAT_GREEN=${TOY_GRASP_RETREAT_GREEN:-}
export TOY_GRASP_RETREAT_BLUE=${TOY_GRASP_RETREAT_BLUE:-}
export TOY_GRASP_RETREAT_RED=${TOY_GRASP_RETREAT_RED:-}
export TOY_GRASP_Z_CLOSE_GREEN=${TOY_GRASP_Z_CLOSE_GREEN:-}
export TOY_GRASP_Z_CLOSE_BLUE=${TOY_GRASP_Z_CLOSE_BLUE:-}
export TOY_GRASP_Z_CLOSE_RED=${TOY_GRASP_Z_CLOSE_RED:-}
export TOY_GRASP_YAW_JITTER=${TOY_GRASP_YAW_JITTER:-0.15}
export TOY_STRICT_GATE_XY_TOL=${TOY_STRICT_GATE_XY_TOL:-0.025}
export TOY_STRICT_GATE_Z_TOL=${TOY_STRICT_GATE_Z_TOL:-0.025}
export TOY_STRICT_GATE_EXTRA_STEPS=${TOY_STRICT_GATE_EXTRA_STEPS:-180}
export TOY_GRASP_HOLD_STEPS=${TOY_GRASP_HOLD_STEPS:-45}

if [[ ! -f "$OBJECT_POSES" ]]; then
  echo "[clear-fsm] Object pose file not found: $OBJECT_POSES" >&2
  echo "[clear-fsm] Create one with:" >&2
  echo "  python cmd_eet/filter_object_poses.py --input data/merge/object_poses.json --output data/object_poses_clear.json --min-distance 0.22 --min-pair-distance 0.12
  python cmd_eet/augment_object_poses.py --input data/object_poses_clear.json --output data/object_poses_clear_aug.json --copies 5 --jitter-xy 0.04" >&2
  exit 2
fi

for i in $(seq 1 "$RUNS"); do
  echo "=== Clear clean FSM run $i / $RUNS ==="
  ARGS=()
  if [[ "$RESUME" == "1" || "$i" != "1" ]]; then
    ARGS+=(--resume)
  fi

  python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id "${HF_USER}/${DATASET_NAME}" \
    --object_poses "$OBJECT_POSES" \
    "${ARGS[@]}"
done
