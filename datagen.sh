export DISPLAY=:4
export HF_USER=ann0000000

# Run 1: fresh start
rm -rf ~/.cache/huggingface/lerobot/${HF_USER}/aic-finalproject-dataset
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --resume \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/aic-finalproject-dataset \
    --object_poses data/merge/object_poses.json

# Run 2: resume from local cache (no HF download)
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --resume \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/aic-finalproject-dataset \
    --object_poses data/merge/object_poses.json

# Run 3: resume from local cache (no HF download)
python scripts/datagen/generate.py \
    --task HCIS-ToyBlocksCollection-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --resume \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/aic-finalproject-dataset \
    --object_poses data/merge/object_poses.json

# Upload once at the end
hf upload ${HF_USER}/aic-finalproject-dataset \
    ~/.cache/huggingface/lerobot/${HF_USER}/aic-finalproject-dataset \
    --repo-type dataset
