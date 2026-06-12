# BC validation harness

This folder contains experiment scripts for behavior-cloning model selection.

- `train_validate.sh` trains one or more LeRobot policies on episode-level splits, then computes supervised held-out loss.
- `split_episodes.py` creates IID (`random`) or covariate-shift-style (`tail`, `length`) splits.
- `validate_lerobot_loss.py` loads a saved `pretrained_model` checkpoint and evaluates `policy.forward(batch)` loss on selected episodes.
- `eval.sh` runs the selected checkpoint in Isaac Lab with `scripts/rollout.py`.

Typical runs:

```bash
# IID overfit check: train loss vs random held-out validation loss.
POLICIES="diffusion act" SPLIT_MODE=random FOLDS="0 1 2" STEPS=50000 bash cmd_eet/train_validate.sh

# Covariate-shift check: train on earlier episodes, validate on latest episodes.
POLICIES=diffusion SPLIT_MODE=tail VAL_FRACTION=0.2 STEPS=50000 bash cmd_eet/train_validate.sh

# Sim rollout for one trained checkpoint.
RUN_GROUP=eet-bc-random-0603-1200 POLICY=diffusion FOLD=0 bash cmd_eet/eval.sh
```

Interpreting results:

- Low train loss and much higher random validation loss means ordinary overfit.
- Similar random validation loss but high `tail`/`length` validation loss means covariate shift.
- Good supervised validation but poor rollout means compounding-error/covariate-shift at execution time; prefer more diverse datagen, stronger visual augmentation, or a chunked/generative policy.

## Early stopping

Use `train_early_stop.sh` when you want to stop training once held-out BC loss stops improving:

```bash
POLICIES=diffusion \
SPLIT_MODE=random \
K_FOLDS=5 \
FOLDS=0 \
MAX_STEPS=100000 \
CHUNK_STEPS=10000 \
PATIENCE=3 \
MIN_DELTA=0.0 \
CUDA_VISIBLE_DEVICES=0 \
bash cmd_eet/train_early_stop.sh
```

What it does:

1. Creates an 80/20 episode split (`K_FOLDS=5`, `FOLDS=0`).
2. Trains until `CHUNK_STEPS`.
3. Computes validation loss on the held-out episodes.
4. Resumes training for another chunk if validation improves or patience remains.
5. Stops once validation has failed to improve for `PATIENCE` validation rounds.

Outputs land under `checkpoints/<run-group>/`:

- `early_stop_results.jsonl` — one validation row per checkpoint interval.
- `summary.txt` — compact table with policy, split, step, and loss.
- `logs/<policy>/fold<k>/early_stop.log` — stop/improvement decisions.
- `logs/<policy>/fold<k>/best_checkpoint.txt` — best validation step, loss, and checkpoint path.
- `<policy>/fold<k>/checkpoints/last/pretrained_model/` — latest checkpoint.

For Isaac Lab rollout validation, first enter the container with `make launch-isaaclab`, then run `bash cmd_eet/eval.sh` with `CHECKPOINT_PATH` set to the best checkpoint from `best_checkpoint.txt`.
