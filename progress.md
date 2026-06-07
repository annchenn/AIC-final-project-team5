# Project Progress

This file is only for concise project progress notes.

## 2026-06-07

- Current focus: verify `HCIS-MovingCubeGrasp-SingleArm-v0` rollout with `jitter=0`.
- `jitter=0` means cube velocity has no extra angular noise; direction still depends on each sampled start position and points toward the workspace center.
- Fixed rollout metrics so only the `success` termination is counted as success; other terminations, such as `cube_fallen`, are counted as failed episodes.
- Previous observed issue: during rollout the cube can slow near the gripper, likely from physical contact. FSM datagen also stopped cube velocity injection at grasp phase, so the expert demos may have included a slowdown before closing.
- Updated FSM datagen to keep injecting cube velocity until the cube is lifted above `cube_lift_threshold`, matching the rollout environment's continuous sliding target. Next verification: run a short datagen smoke test in IsaacLab before regenerating the dataset.
