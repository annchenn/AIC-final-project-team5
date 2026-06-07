# Project Progress

This file is only for concise project progress notes.

## 2026-06-07

- Current focus: verify `HCIS-MovingCubeGrasp-SingleArm-v0` rollout with `jitter=0`.
- `jitter=0` means cube velocity has no extra angular noise; direction still depends on each sampled start position and points toward the workspace center.
- Fixed rollout metrics so only the `success` termination is counted as success; other terminations, such as `cube_fallen`, are counted as failed episodes.
- Observed issue: during rollout the cube can slow near the gripper, likely from physical contact. In FSM datagen, cube velocity injection stops at grasp phase, so the expert demos may include a slowdown before closing.
