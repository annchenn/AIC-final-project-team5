#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export HF_USER=${HF_USER:-ann0000000}
export DATASET_NAME=${DATASET_NAME:-aic-finalproject-dataset-eettest}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/workspace/aicapstone/datasets/lerobot_cache}
export OBJECT_POSES=${OBJECT_POSES:-}
export RANDOM_OBJECT_POSES=${RANDOM_OBJECT_POSES:-60}
export RANDOM_POSE_MODE=${RANDOM_POSE_MODE:-eval_jitter}
export RANDOM_POSE_JITTER_XY=${RANDOM_POSE_JITTER_XY:-0.05}
export RANDOM_POSE_X_RANGE=${RANDOM_POSE_X_RANGE:-0.05,0.55}
export RANDOM_POSE_Y_RANGE=${RANDOM_POSE_Y_RANGE:--0.62,-0.12}
if [[ -z "${RANDOM_POSE_MIN_BASKET_DIST+x}" ]]; then
    if [[ "$RANDOM_POSE_MODE" == "eval_jitter" ]]; then
        export RANDOM_POSE_MIN_BASKET_DIST=0.0
    else
        export RANDOM_POSE_MIN_BASKET_DIST=0.24
    fi
fi
if [[ -z "${RANDOM_POSE_MIN_PAIR_DIST+x}" ]]; then
    if [[ "$RANDOM_POSE_MODE" == "eval_jitter" ]]; then
        export RANDOM_POSE_MIN_PAIR_DIST=0.0
    else
        export RANDOM_POSE_MIN_PAIR_DIST=0.13
    fi
fi
export RANDOM_POSE_BASKET_XY=${RANDOM_POSE_BASKET_XY:-0.65,-0.55}
export RANDOM_POSE_YAW_MODE=${RANDOM_POSE_YAW_MODE:-fixed}
export RUNS=${RUNS:-1}
export FRESH=${FRESH:-0}
export DATAGEN_EPISODE_TIMEOUT_S=${DATAGEN_EPISODE_TIMEOUT_S:-240}
export TOY_FSM_DEBUG_INTERVAL=${TOY_FSM_DEBUG_INTERVAL:-30}
export TOY_FSM_MAX_CARTESIAN_DELTA=${TOY_FSM_MAX_CARTESIAN_DELTA:-0.024}
export TOY_FSM_HOVER_Z_OFFSET=${TOY_FSM_HOVER_Z_OFFSET:-0.45}
export TOY_FSM_LIFT_Z_OFFSET=${TOY_FSM_LIFT_Z_OFFSET:-0.16}
export TOY_FSM_BOX_HOVER_Z_OFFSET=${TOY_FSM_BOX_HOVER_Z_OFFSET:-0.40}
export TOY_FSM_RELEASE_Z_OFFSET=${TOY_FSM_RELEASE_Z_OFFSET:-0.09}
export TOY_FSM_HOVER_XY_TOL=${TOY_FSM_HOVER_XY_TOL:-0.012}
export TOY_FSM_HOVER_Z_TOL=${TOY_FSM_HOVER_Z_TOL:-0.080}
export TOY_FSM_APPROACH_XY_TOL=${TOY_FSM_APPROACH_XY_TOL:-0.045}
export TOY_FSM_APPROACH_Z_TOL=${TOY_FSM_APPROACH_Z_TOL:-0.025}
export TOY_FSM_GRASP_XY_TOL=${TOY_FSM_GRASP_XY_TOL:-0.025}
export TOY_FSM_GRASP_Z_TOL=${TOY_FSM_GRASP_Z_TOL:-0.025}
export TOY_FSM_LIFT_XY_TOL=${TOY_FSM_LIFT_XY_TOL:-0.040}
export TOY_FSM_LIFT_Z_TOL=${TOY_FSM_LIFT_Z_TOL:-0.025}
export TOY_FSM_TO_BOX_XY_TOL=${TOY_FSM_TO_BOX_XY_TOL:-0.050}
export TOY_FSM_TO_BOX_Z_TOL=${TOY_FSM_TO_BOX_Z_TOL:-0.080}
export TOY_FSM_RETREAT_XY_TOL=${TOY_FSM_RETREAT_XY_TOL:-0.060}
export TOY_FSM_RETREAT_Z_TOL=${TOY_FSM_RETREAT_Z_TOL:-0.080}
export TOY_FSM_GRIPPER_CLOSED_WIDTH=${TOY_FSM_GRIPPER_CLOSED_WIDTH:-0.045}
export TOY_FSM_GRASP_SETTLE_STEPS=${TOY_FSM_GRASP_SETTLE_STEPS:-25}
export TOY_FSM_DROP_SETTLE_STEPS=${TOY_FSM_DROP_SETTLE_STEPS:-90}
export TOY_FSM_PHASE_TIMEOUT_STEPS=${TOY_FSM_PHASE_TIMEOUT_STEPS:-1500}
export TOY_FSM_IK_REPANIC_AFTER_STEPS=${TOY_FSM_IK_REPANIC_AFTER_STEPS:-300}
export TOY_FSM_IK_REPANIC_STEPS=${TOY_FSM_IK_REPANIC_STEPS:-90}
export TOY_FSM_IK_REPANIC_X=${TOY_FSM_IK_REPANIC_X:-0.5}
export TOY_FSM_IK_REPANIC_Y=${TOY_FSM_IK_REPANIC_Y:--0.5}
export TOY_FSM_IK_REPANIC_Z=${TOY_FSM_IK_REPANIC_Z:-0.4}
export TOY_FSM_DEFAULT_OBJECT_HEIGHT=${TOY_FSM_DEFAULT_OBJECT_HEIGHT:-0.060}
export TOY_FSM_APPROACH_Z_HEIGHT_FRAC=${TOY_FSM_APPROACH_Z_HEIGHT_FRAC:-0.45}
export TOY_FSM_GRASP_Z_HEIGHT_FRAC=${TOY_FSM_GRASP_Z_HEIGHT_FRAC:-0.10}
export TOY_FSM_APPROACH_Z_READY_HEIGHT_FRAC=${TOY_FSM_APPROACH_Z_READY_HEIGHT_FRAC:-1.25}
export TOY_FSM_GRASP_Z_READY_HEIGHT_FRAC=${TOY_FSM_GRASP_Z_READY_HEIGHT_FRAC:-1.25}
export TOY_FSM_APPROACH_Z_READY_ABS=${TOY_FSM_APPROACH_Z_READY_ABS:-0.080}
export TOY_FSM_GRASP_Z_READY_ABS=${TOY_FSM_GRASP_Z_READY_ABS:-0.100}
export TOY_FSM_BASKET_CLEARANCE_Z=${TOY_FSM_BASKET_CLEARANCE_Z:-0.060}
export TOY_FSM_CARRY_Z_ABS=${TOY_FSM_CARRY_Z_ABS:-0.45}
export TOY_FSM_USE_BBOX_CENTER=${TOY_FSM_USE_BBOX_CENTER:-1}
export TOY_FSM_BBOX_CENTER_Z_OFFSET=${TOY_FSM_BBOX_CENTER_Z_OFFSET:-0.0}
export TOY_FSM_FIRST_BLOCK_GRASP_X_BIAS=${TOY_FSM_FIRST_BLOCK_GRASP_X_BIAS:--0.01}
export TOY_FSM_SECOND_BLOCK_GRASP_X_BIAS=${TOY_FSM_SECOND_BLOCK_GRASP_X_BIAS:--0.015}
export TOY_FSM_LAST_BLOCK_GRASP_X_BIAS=${TOY_FSM_LAST_BLOCK_GRASP_X_BIAS:-0.020}

DATASET_DIR="${HF_LEROBOT_HOME}/${HF_USER}/${DATASET_NAME}"

if [[ "$FRESH" == "1" ]]; then
    echo "[datagen] Removing local dataset: ${DATASET_DIR}"
    rm -rf "$DATASET_DIR"
fi

if [[ -d "$DATASET_DIR" && ! -f "$DATASET_DIR/meta/tasks.parquet" ]]; then
    echo "[datagen] Broken local dataset exists: ${DATASET_DIR}" >&2
    echo "[datagen] It is missing meta/tasks.parquet, so LeRobot may try Hugging Face." >&2
    echo "[datagen] Use FRESH=1 to delete/recreate it, or choose a new DATASET_NAME." >&2
    exit 2
fi

for i in $(seq 1 "$RUNS"); do
    echo "=== Run $i / $RUNS ==="

    ARGS=()
    if [[ -f "$DATASET_DIR/meta/tasks.parquet" ]]; then
        ARGS+=(--resume)
    fi

    POSE_ARGS=()
    if [[ -n "$OBJECT_POSES" ]]; then
        POSE_ARGS+=(--object_poses "$OBJECT_POSES")
    else
        POSE_ARGS+=(
            --random_object_poses "$RANDOM_OBJECT_POSES"
            --target_success_count "$RANDOM_OBJECT_POSES"
            --random_pose_mode "$RANDOM_POSE_MODE"
            --random_pose_jitter_xy "$RANDOM_POSE_JITTER_XY"
            --random_pose_x_range="$RANDOM_POSE_X_RANGE"
            --random_pose_y_range="$RANDOM_POSE_Y_RANGE"
            --random_pose_min_basket_dist="$RANDOM_POSE_MIN_BASKET_DIST"
            --random_pose_min_pair_dist="$RANDOM_POSE_MIN_PAIR_DIST"
            --random_pose_basket_xy="$RANDOM_POSE_BASKET_XY"
            --random_pose_yaw_mode="$RANDOM_POSE_YAW_MODE"
        )
    fi

    python scripts/datagen/generate.py \
        --task HCIS-ToyBlocksCollection-SingleArm-v0 \
        --num_envs 1 \
        --device cuda \
        --enable_cameras \
        --record \
        --use_lerobot_recorder \
        --lerobot_dataset_repo_id "${HF_USER}/${DATASET_NAME}" \
        "${POSE_ARGS[@]}" \
        --episode_timeout_s "$DATAGEN_EPISODE_TIMEOUT_S" \
        "${ARGS[@]}"
done
