# AI Capstone

Sim-to-real imitation-learning pipeline for robot manipulation tasks. Record human demonstrations with UMI, process them through SLAM, generate synthetic data in Isaac Lab, train a LeRobot policy, and evaluate it in simulation. This branch adds temporal observations, SmolVLA training, and validation-loss early stopping for the moving-cube task.

> **Platform:** Linux only.

For a complete step-by-step walkthrough, see [Getting Started](docs/getting_started.md).

# Human Demonstration Data Processing

1. **Installation**

   ```bash
   uv sync --package umi
   ```

2. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate
   ```

   This makes `hf`, `lerobot-train`, and other installed commands available in your terminal.

3. **Hugging Face login**

   Create an access token at: <https://huggingface.co/docs/hub/en/security-tokens>

   Then log in:

   ```bash
   hf auth login --token <YOUR_HF_TOKEN>
   ```

4. **Set your Hugging Face username**

   Commands throughout this project use `${HF_USER}`. Set it once per terminal session:

   ```bash
   export HF_USER=<your-huggingface-username>
   ```

## After recording the demonstration videos, follow this practice

1. Under `data/`, create a directory for this demo. Suggested name: `YYYYMMDD-taskname`. Add a `raw_videos/` subdirectory under it.
2. Place the recorded videos in `data/YYYYMMDD-taskname/raw_videos/`.

## Verify the recorded demonstration videos

The SLAM mapping stage is fragile. To save time, run the verify pipeline first:

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline.yaml \
    --session-dir <demo_directory_name>
```

## If verification fails, re-record and copy into the demo directory

There are several failure modes:

### SLAM failures

Pipeline raises:

```
RuntimeError: SLAM mapping failed. Check logs at datasets/team_asia/demos/mapping/slam_stdout.txt for details.
```

Re-record the mapping video, replace the file, and re-run the verification pipeline.

## If verification succeeds, run the full pipeline

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset.yaml \
    --session-dir <demo_directory_name> \
    --task <kitchen|dining_room|living_room>
```

Upload the whole session directory to the Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json
```

# Data Creation in Simulator

## Prerequisites

1. **Linux machine with Nvidia GPU** — verify with `nvidia-smi`. Isaac Lab requires a Linux host with an Nvidia driver.
2. **Docker installed** — the simulator runs inside a container.
3. **Repository cloned** — if you haven't already:
   ```bash
   git clone https://github.com/HCIS-Lab/aicapstone.git
   cd aicapstone
   ```

## Launch Isaac Lab

```bash
make launch-isaaclab
```

This builds the Isaac Sim container. On success, the shell drops you inside the container.

Download the session directory produced by the UMI pipeline:

```bash
hf download ${HF_USER}/<repo_id> --local-dir data/<demo_directory_name>
```

## Run the data generation pipeline

The `--lerobot_dataset_repo_id` should be your own Hugging Face dataset repo.

Available tasks:

- `HCIS-CupStacking-SingleArm-v0`
- `HCIS-CutleryArrangement-SingleArm-v0`
- `HCIS-ToyBlocksCollection-SingleArm-v0`

```bash
python scripts/datagen/generate.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
    --object_poses data/<demo_directory_name>/object_poses.json
```

Upload the recorded dataset to Hugging Face Hub:

```bash
hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/
```

# LeRobot Training

Training runs on the **host machine** (not inside Docker) and produces a policy checkpoint from your generated dataset. Requires an Nvidia GPU.

## SmolVLA dependencies

The temporal SmolVLA workflow uses the SmolVLA implementation already included
with LeRobot. Compared with the original diffusion/ACT workflow, it additionally
requires:

- `transformers>=4.57.6` for the SmolVLM2 vision-language backbone.
- `num2words>=0.5.14` for the SmolVLM processor.
- The pretrained `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` model files. These
  are downloaded from Hugging Face when first needed and are not a Python package.

Install the host dependencies from the repository root:

```bash
uv sync
```

The Isaac Lab Docker image installs the same Python dependencies during build.
If using an older existing image that reports that `num2words` is missing,
install it once inside that running container:

```bash
python -m pip install "num2words>=0.5.14"
```

Containers started with `--rm` lose manual package installations when they exit.
Rebuild the image to make the dependency permanent:

```bash
make build-isaaclab
```

## Temporal SmolVLA with early stopping

Run training on the host machine, not inside the Isaac Lab container. The
following commands use the moving-cube dataset and the defaults from
`cmd/train_moving_cube_smolvla_temporal_early_stop.sh`.

### 1. Download the dataset

Log in to Hugging Face once, then download the complete LeRobot dataset into
the path expected by the training script:

```bash
hf auth login

export DATASET_OWNER=yun0523
export DATASET_NAME=moving-cube-grasp
export DATASET_ROOT="$PWD/datasets/lerobot_cache/${DATASET_OWNER}/${DATASET_NAME}"

hf download "${DATASET_OWNER}/${DATASET_NAME}" \
  --repo-type dataset \
  --local-dir "${DATASET_ROOT}"
```

Log in to W&B for online training metrics:

```bash
uv run wandb login
```

After downloading, this file must exist:

```text
datasets/lerobot_cache/yun0523/moving-cube-grasp/meta/info.json
```

### 2. Start training

This example validates every 20,000 optimizer steps. `PATIENCE=1` stops after
the first validation that does not improve on the best previous validation.
`VAL_MAX_BATCHES=500` limits validation time; set it to `0` to use every
validation example.

```bash
SOURCE_DATASET_ROOT="${DATASET_ROOT}" \
MAX_STEPS=200000 \
EVAL_EVERY=20000 \
PATIENCE=1 \
MIN_DELTA=0 \
VAL_MAX_BATCHES=500 \
VAL_BATCH_SIZE=8 \
WANDB_ENABLE=true \
bash cmd/train_moving_cube_smolvla_temporal_early_stop.sh \
  moving-cube-temporal-es-v1
```

The first run automatically creates a fixed train/validation episode split.
The main outputs are:

```text
checkpoints/moving-cube-temporal-es-v1/checkpoints/020000/
checkpoints/moving-cube-temporal-es-v1_early_stop/episode_split.json
checkpoints/moving-cube-temporal-es-v1_early_stop/early_stop_state.json
checkpoints/moving-cube-temporal-es-v1_early_stop/validation_history.jsonl
```

To continue an interrupted run, run the same command again with the same run
name. The script finds the highest numbered complete checkpoint and restores
the optimizer, scheduler, training step, and earlier validation history.

### 3. Download an existing checkpoint

Training checkpoints are separate from the SmolVLM2 backbone downloaded by
Transformers. To evaluate or resume a checkpoint already uploaded by the team,
download its model repository into the same local run directory:

```bash
export MODEL_REPO=mikehsuhoodie/aic-finalproject-team5-smolvla-temporal
export RUN_NAME=moving-cube-temporal-es-v1

hf download "${MODEL_REPO}" \
  --local-dir "checkpoints/${RUN_NAME}"
```

For resuming training, the downloaded checkpoint must include both
`pretrained_model/` and `training_state/`. A directory containing only
`pretrained_model/` is sufficient for rollout, but cannot restore the optimizer
or continue from the saved training step.

## Train with a teammate dataset

If you want to train with a dataset that someone else already uploaded, set the
dataset owner separately from your own Hugging Face username. The dataset owner
is the account that owns the uploaded dataset. Your own username is used for the
new policy output, so you do not overwrite another teammate's training result.

For example, to train from Mike's moving-cube dataset:

```bash
export DATASET_OWNER=mikehsuhoodie
export DATASET_NAME=moving-cube-grasp-fsm-20260607-mike
export HF_USER=<your-huggingface-username>
export RUN_NAME=moving-cube-grasp-from-mike-data-<your-name>
export POLICY_REPO=aic-finalproject-team5-<your-name>
```

Then train on the host machine:

```bash
lerobot-train \
  --dataset.repo_id=${DATASET_OWNER}/${DATASET_NAME} \
  --dataset.root=datasets/lerobot_cache \
  --policy.type=diffusion \
  --output_dir=checkpoints/${RUN_NAME} \
  --job_name=${RUN_NAME} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/${POLICY_REPO}
```

Use a unique `RUN_NAME` and `POLICY_REPO` for each training run. This keeps the
input dataset shared while keeping each person's trained checkpoint separate.

See [LeRobot Training Procedure](docs/lerobot_training.md) for the full command reference, multi-GPU setup, and troubleshooting.


# LeRobot Rollout

Rollout loads your trained policy into the Isaac Lab simulator (inside the Docker container) to evaluate robot performance.

`scripts/rollout.py` runs the policy in the simulator and is the right tool for
watching the robot, debugging one checkpoint, or testing a chosen cube speed.
`eval/moving_cube_grasp_eval.py` automates repeated rollouts at several speeds,
collects capture, placement, and end-to-end success rates, and generates a
summary plot. The current eval wrapper still selects the original diffusion
policy, so use `scripts/rollout.py` for temporal SmolVLA checkpoints until that
wrapper is updated.

## Temporal SmolVLA rollout notes

SmolVLA rollout differs from the original diffusion-policy workflow in several
important ways:

- Use `--policy_type lerobot-smolvla`.
- Point `--policy_checkpoint_path` at the checkpoint's `pretrained_model/`
  directory, not the parent checkpoint directory.
- Keep `--policy_action_horizon 4`, matching the value used for training.
- Keep `--policy_observation_fps 30` so the temporal camera history is sampled
  at the same rate expected by the policy.
- Use `scripts/rollout.py` directly. The current
  `eval/moving_cube_grasp_eval.py` wrapper is for diffusion policies.
- `--enable_cameras` is required. `--headless` hides the simulator window but
  still renders the wrist and front camera observations used by SmolVLA.

On a regular SSH server with an X server available at display `:0`, launch the
container from the repository root on the host:

```bash
DISPLAY=:0 make launch-isaaclab
```

Then run the 60,000-step checkpoint from inside the container:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

python scripts/rollout.py \
  --headless \
  --task HCIS-MovingCubeGrasp-SingleArm-v0 \
  --policy_type lerobot-smolvla \
  --policy_checkpoint_path checkpoints/moving-cube-temporal-es-v1/checkpoints/060000/pretrained_model \
  --policy_action_horizon 4 \
  --policy_observation_fps 30 \
  --eval_rounds 20 \
  --episode_length_s 30 \
  --device cuda \
  --enable_cameras
```

VNC is not required for this headless command. Before accepting the rollout
results, check the startup log. It should list the Nvidia GPU with
`Graphics API: Vulkan`. Do not use results from a run that reports
`ERROR_INCOMPATIBLE_DRIVER`, `no suitable CUDA GPU`, or that the PhysX GPU
pipeline switched to software.

See [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) for the full procedure.

## AI-Facing Notes

These root Markdown files are used by AI assistants and collaborators:

| File | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Instructions for AI coding agents: project constraints, package manager, and coding style. |
| [ClAUDE.md](ClAUDE.md) | Course/project direction and proposal context. Use it for goals, requirements, and evaluation targets. |
| [spec.md](spec.md) | Project specification and submission requirements. Treat it as requirements, not progress. |
| [advanced_level.md](advanced_level.md) | Advanced task design and implementation summary. Keep it stable and report-oriented. |
| `hackmd.md` | Optional raw team working notes copied from HackMD. |


## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting_started.md) | End-to-end pipeline walkthrough |
| [Developer Introduction](docs/dev/introduction.md) | Repo layout, environment setup, where to run what |
| [Isaac Lab + LeIsaac Configuration Tutorial](docs/isaaclab_leisaac_tutorial.md) | Configuring Isaac Lab with LeIsaac |
| [LeRobot Dataset Visualizer](docs/lerobot_dataset_visualizer.md) | Visualizing LeRobot datasets |
| [LeRobot Checkpoint Format](docs/lerobot_model_format.md) | Understanding LeRobot model checkpoint structure |
| [LeRobot Rollout (Policy Evaluation)](docs/lerobot_rollout.md) | Running trained policies in the simulator |
| [LeRobot Training Procedure](docs/lerobot_training.md) | Training imitation-learning policies |
| [Standalone Env Config Export](docs/standalone_env_config_export.md) | Exporting environment configs as standalone files |
| [Synthetic Data Generation Pipeline](docs/synthetic_data_generation.md) | Generating synthetic training data |
| [UMI Pipeline](docs/umi_pipeline.md) | Data collection and processing with UMI |

## License

MIT — see [LICENSE](LICENSE).
