export DISPLAY=:4
export HF_USER=ann0000000
export DATASET_NAME=aic-finalproject-dataset-v2
export HF_LEROBOT_HOME=/workspace/aicapstone/datasets/lerobot_cache

# # Run 1: fresh start
# rm -rf ${HF_LEROBOT_HOME}/${HF_USER}/${DATASET_NAME}
# python scripts/datagen/generate.py \
#     --task HCIS-ToyBlocksCollection-SingleArm-v0 \
#     --num_envs 1 \
#     --device cuda \
#     --enable_cameras \
#     --record \
#     --use_lerobot_recorder \
#     --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
#     --object_poses data/merge/object_poses.json

for i in $(seq 1 5); do
    echo "=== Run $i / 5 ==="
    python scripts/datagen/generate.py \
        --task HCIS-ToyBlocksCollection-SingleArm-v0 \
        --num_envs 1 \
        --device cuda \
        --enable_cameras \
        --record \
        --resume \
        --use_lerobot_recorder \
        --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
        --object_poses data/merge/object_poses.json
done

# # Upload once at the end
# hf upload ${HF_USER}/aic-finalproject-dataset \
#     ~/.cache/huggingface/lerobot/${HF_USER}/aic-finalproject-dataset \
#     --repo-type dataset
