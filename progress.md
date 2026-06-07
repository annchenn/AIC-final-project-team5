# Project Progress

This file is only for concise project progress notes.

## 2026-06-07

- Current focus: verify `HCIS-MovingCubeGrasp-SingleArm-v0` rollout with `jitter=0`.
- `jitter=0` means cube velocity has no extra angular noise; direction still depends on each sampled start position and points toward the workspace center.
- Fixed rollout metrics so only the `success` termination is counted as success; other terminations, such as `cube_fallen`, are counted as failed episodes.
- Previous observed issue: during rollout the cube can slow near the gripper, likely from physical contact. FSM datagen also stopped cube velocity injection at grasp phase, so the expert demos may have included a slowdown before closing.
- Updated FSM datagen to keep injecting cube velocity until the cube is lifted above `cube_lift_threshold`, matching the rollout environment's continuous sliding target. Next verification: run a short datagen smoke test in IsaacLab before regenerating the dataset.
- Regenerated the moving-cube LeRobot dataset as mikehsuhoodie/moving-cube-grasp-fsm-20260607-mike; observed dataset size is 78 successful episodes and 48,750 frames from 80 planned synthetic cube poses.
- Trained diffusion policy run moving-cube-grasp-fsm-20260607-mike-run5 on the host machine. Local checkpoints exist at 20k, 40k, 60k, 80k, and 100k steps under checkpoints/moving-cube-grasp-fsm-20260607-mike-run5/checkpoints/.
- Added README instructions for training from a teammate dataset without overwriting that teammate policy outputs.
- Rollout/eval next step: download the selected checkpoint revision in the IsaacLab container and run scripts/rollout.py or eval/moving_cube_grasp_eval.py with jitter=0. On the L40S instance, Isaac GUI display is blocked by incomplete VirtualGL; vglrun glxinfo reports llvmpipe, so use --headless or terminal metrics until the GUI stack is fixed.
