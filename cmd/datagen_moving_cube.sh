export HF_USER=yun0523
export DATASET_NAME=aic-finalproject-moving-cube-v1
export HF_LEROBOT_HOME=/workspace/aicapstone/datasets/lerobot_cache
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

mkdir -p ${HF_LEROBOT_HOME}

# Run 1: fresh start
rm -rf ${HF_LEROBOT_HOME}/${HF_USER}/${DATASET_NAME}
python scripts/datagen/generate.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
    --object_poses data/moving_cube_demo/object_poses.json

# # Run 2+: resume (uncomment if success rate < 100%)
# for i in $(seq 1 5); do
#     echo "=== Resume Run $i / 5 ==="
#     python scripts/datagen/generate.py \
#         --task HCIS-MovingCubeGrasp-SingleArm-v0 \
#         --num_envs 1 \
#         --device cuda \
#         --enable_cameras \
#         --record \
#         --resume \
#         --use_lerobot_recorder \
#         --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
#         --object_poses data/moving_cube_demo/object_poses.json
# done
