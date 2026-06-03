export HF_USER=yun0523
export DATASET_NAME=aic-finalproject-moving-cube-v1
export RUN_NAME=${1:-moving_cube_run1}
mkdir -p ~/tmp ~/wandb
export TMPDIR=~/tmp
export WANDB_DIR=~/wandb
export WANDB_CACHE_DIR=~/wandb
export HF_HUB_OFFLINE=1

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

lerobot-train \
  --dataset.repo_id=${HF_USER}/${DATASET_NAME} \
  --dataset.root=${SCRIPT_DIR}/datasets/lerobot_cache/${HF_USER}/${DATASET_NAME} \
  --policy.type=diffusion \
  --output_dir=${SCRIPT_DIR}/checkpoints/${RUN_NAME} \
  --job_name=${RUN_NAME} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/aic-finalproject-team5
