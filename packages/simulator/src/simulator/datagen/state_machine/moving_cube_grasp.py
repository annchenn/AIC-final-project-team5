"""State machine for the Advanced-level moving-cube grasping task.

Reuses the Franka IK / pose helpers from ``ToyBlocksCollectionStateMachine``
and adds:

* ``pre_step``: writes a constant linear velocity to the cube each tick so
  it slides across the table until the gripper closes.
* a predictive ``get_action`` that simply tracks the (moving) cube position —
  the per-step Cartesian delta clamp (~1 m/s at 60 Hz) is an order of magnitude
  faster than the cube's drift, so direct pursuit is sufficient.

Phases per episode (single object):
    0. hover above cube
    1. approach down (tracking moving target)
    2. close gripper
    3. lift
"""

from __future__ import annotations

import math

import torch

from simulator.datagen.state_machine.toy_blocks_collection import (
    ToyBlocksCollectionStateMachine,
    _GRASP_YAW_OFFSET,
    _GRASP_Z_OFFSET,
    _GRIPPER_CLOSE,
    _GRIPPER_OPEN,
    _HOVER_Z_OFFSET,
    _LIFT_Z_OFFSET,
    _constant_gripper,
)

_CUBE_NAME = "cube"
# (hover, approach, grasp_close, lift). 60 Hz step rate.
_PHASE_DURATIONS = (120, 140, 20, 80)
_PHASES = len(_PHASE_DURATIONS)


class MovingCubeGraspStateMachine(ToyBlocksCollectionStateMachine):
    """Chase + grasp + lift a sliding cube."""

    MAX_STEPS: int = sum(_PHASE_DURATIONS) + 60

    def __init__(self) -> None:
        super().__init__()
        # Override the multi-object event timeline with our 4 phases.
        self._events_dt = list(_PHASE_DURATIONS)
        self._cube_velocity_w: torch.Tensor | None = None  # (num_envs, 6)
        self._speed_range: tuple[float, float] = (0.05, 0.10)
        self._lift_threshold: float = 0.12

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        super().setup(env)
        # Read optional knobs off the env cfg (set by MovingCubeGraspEnvCfg).
        cfg = getattr(env, "cfg", None)
        if cfg is not None:
            self._speed_range = tuple(getattr(cfg, "cube_linear_speed_range", self._speed_range))
            self._lift_threshold = float(getattr(cfg, "cube_lift_threshold", self._lift_threshold))
        self._sample_cube_velocity(env)

    def reset(self) -> None:
        super().reset()
        self._cube_velocity_w = None  # resampled on next pre_step

    def _sample_cube_velocity(self, env) -> None:
        """Pick a random horizontal velocity for the cube (per env)."""
        device = env.device
        num_envs = env.num_envs
        speed = torch.empty(num_envs, device=device).uniform_(*self._speed_range)
        # Direction: random angle in the table plane, biased toward +y (away
        # from the robot base) so the cube does not slide off the workspace.
        angle = torch.empty(num_envs, device=device).uniform_(math.pi / 4.0, 3.0 * math.pi / 4.0)
        vx = speed * torch.cos(angle)
        vy = speed * torch.sin(angle)
        vz = torch.zeros_like(vx)
        zero = torch.zeros_like(vx)
        # (vx, vy, vz, wx, wy, wz)
        self._cube_velocity_w = torch.stack([vx, vy, vz, zero, zero, zero], dim=-1)

    # ------------------------------------------------------------------
    # Per-step velocity injection
    # ------------------------------------------------------------------

    def pre_step(self, env) -> None:
        if self._cube_velocity_w is None:
            self._sample_cube_velocity(env)
            vx0 = float(self._cube_velocity_w[0, 0].item())
            vy0 = float(self._cube_velocity_w[0, 1].item())
            print(f"[moving_cube] new episode cube velocity: vx={vx0:+.3f} vy={vy0:+.3f} m/s")

        # Stop driving the cube once we have started closing the gripper —
        # from phase 2 onward the cube should follow physics (grasped or not).
        if self._event >= 2:
            return

        cube = env.scene[_CUBE_NAME]
        # Re-assert horizontal velocity each tick (gravity zeroes vz between
        # steps; we keep vx/vy constant for predictable scripted demos).
        vel = self._cube_velocity_w.to(device=env.device, dtype=torch.float32)
        cube.write_root_velocity_to_sim(vel)

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs

        cube = env.scene[_CUBE_NAME]
        cube_pos_w = cube.data.root_pos_w.clone()
        cube_quat_w = cube.data.root_quat_w.clone()

        if self._step_count == 0 and self._event == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        target_quat_w = self._gripper_down_quat_w(
            cube_quat_w, num_envs, device, cube_quat_w.dtype, yaw_offset=_GRASP_YAW_OFFSET
        )

        phase = self._event
        if phase == 0:
            # Hover above the (moving) cube; smoothly interpolate from rest.
            target = cube_pos_w.clone()
            target[:, 2] += _HOVER_Z_OFFSET
            if self._initial_ee_pos_w is not None:
                denom = max(self._events_dt[self._event] - 1, 1)
                alpha = min(self._step_count / denom, 1.0)
                target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
            gripper = _constant_gripper(num_envs, device, _GRIPPER_OPEN)
        elif phase == 1:
            # Approach: predictive lead based on remaining time × cube velocity.
            target = cube_pos_w.clone()
            target[:, 2] += _GRASP_Z_OFFSET
            if self._cube_velocity_w is not None:
                steps_left = max(self._events_dt[self._event] - self._step_count, 0)
                dt = env.physics_dt * env.cfg.decimation
                lead = self._cube_velocity_w[:, :3].to(target.device, dtype=target.dtype) * (
                    steps_left * dt
                )
                target[:, :2] += lead[:, :2]
            gripper = _constant_gripper(num_envs, device, _GRIPPER_OPEN)
        elif phase == 2:
            # Close gripper at the current cube position (cube now follows physics).
            target = cube_pos_w.clone()
            target[:, 2] += _GRASP_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_CLOSE)
        else:
            # Lift straight up.
            target = cube_pos_w.clone()
            target[:, 2] += _LIFT_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

        return self._joint_position_franka_action(env, target, target_quat_w, gripper)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def check_success(self, env) -> bool:
        cube_pos = env.scene[_CUBE_NAME].data.root_pos_w - env.scene.env_origins
        # rest z ≈ OBJECT_Z (0.05); success if lifted by ``lift_threshold``.
        return bool((cube_pos[:, 2] > (0.05 + self._lift_threshold)).all().item())

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_CUBE_NAME,)
