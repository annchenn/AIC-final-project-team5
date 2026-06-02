"""State machine for the Advanced-level moving-cube-into-basket task.

Reuses the Franka IK / pose / place helpers from
``ToyBlocksCollectionStateMachine`` and adds:

* ``pre_step``: writes a constant linear velocity to the cube each tick so it
  slides across the table until the gripper closes. The velocity direction is
  sampled at episode start so the cube always heads **toward the centre of the
  workspace** (within ±60°). This keeps the trajectory a clean straight line
  for the whole chase phase — no bouncing, no direction changes — which is the
  simplest dynamics for an imitation-learning policy to fit.
* a predictive ``get_action`` that tracks the (moving) cube position with a
  lead term, then carries it to the basket and releases.

Phases per episode (single object):
    0. hover above cube
    1. approach down (tracking moving target)
    2. close gripper
    3. lift
    4. move above basket
    5. lower into basket
    6. release + retreat
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
    _RELEASE_Z_OFFSET,
    _constant_gripper,
)

_CUBE_NAME = "cube"
_BASKET_NAME = "basket"
# (hover, approach, grasp_close, lift, move_above_basket, lower, release/retreat).
# 60 Hz step rate.
_PHASE_DURATIONS = (120, 140, 20, 80, 140, 30, 40)
_PHASES = len(_PHASE_DURATIONS)


class MovingCubeGraspStateMachine(ToyBlocksCollectionStateMachine):
    """Chase + grasp + lift + place a sliding cube into the basket."""

    MAX_STEPS: int = sum(_PHASE_DURATIONS) + 60

    def __init__(self) -> None:
        super().__init__()
        # Override the multi-object event timeline with our 7 phases.
        self._events_dt = list(_PHASE_DURATIONS)
        self._cube_velocity_w: torch.Tensor | None = None  # (num_envs, 6)
        self._speed_range: tuple[float, float] = (0.05, 0.10)
        self._lift_threshold: float = 0.12
        self._workspace_x: tuple[float, float] = (-1.0, 1.0)
        self._workspace_y: tuple[float, float] = (-1.0, 1.0)

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
            self._workspace_x = tuple(getattr(cfg, "cube_workspace_x", self._workspace_x))
            self._workspace_y = tuple(getattr(cfg, "cube_workspace_y", self._workspace_y))
        self._sample_cube_velocity(env)

    def reset(self) -> None:
        super().reset()
        self._cube_velocity_w = None  # resampled on next pre_step

    def _sample_cube_velocity(self, env) -> None:
        """Pick a horizontal velocity that always points toward the workspace centre.

        Per env we compute the bearing from the current cube position to the
        workspace centre and sample the velocity angle within ±60° of that
        bearing. The cube therefore heads inward and travels in a clean
        straight line for the whole chase phase — no bouncing, no direction
        changes — which is the simplest dynamics for an imitation policy to fit.
        """
        device = env.device
        num_envs = env.num_envs

        cube_pos = env.scene[_CUBE_NAME].data.root_pos_w - env.scene.env_origins
        x_min, x_max = self._workspace_x
        y_min, y_max = self._workspace_y
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        dx = cx - cube_pos[:, 0]
        dy = cy - cube_pos[:, 1]
        bearing = torch.atan2(dy, dx)
        jitter = torch.empty(num_envs, device=device).uniform_(-math.pi / 3.0, math.pi / 3.0)
        angle = bearing + jitter

        speed = torch.empty(num_envs, device=device).uniform_(*self._speed_range)
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
        # Re-assert horizontal velocity each tick (friction would otherwise
        # decelerate it). Direction is fixed for the whole episode, so the
        # trajectory stays a clean straight line.
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

        basket = env.scene[_BASKET_NAME]
        basket_pos_w = basket.data.root_pos_w.clone()

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
        elif phase == 3:
            # Lift straight up from where we grasped.
            target = cube_pos_w.clone()
            target[:, 2] += _LIFT_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_CLOSE)
        elif phase == 4:
            # Move above the basket while still holding the cube.
            target = basket_pos_w.clone()
            target[:, 2] += _LIFT_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_CLOSE)
        elif phase == 5:
            # Lower toward the basket interior.
            target = basket_pos_w.clone()
            target[:, 2] += _RELEASE_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_CLOSE)
        else:
            # Open gripper and retreat upward.
            target = basket_pos_w.clone()
            target[:, 2] += _LIFT_Z_OFFSET
            gripper = _constant_gripper(num_envs, device, _GRIPPER_OPEN)

        return self._joint_position_franka_action(env, target, target_quat_w, gripper)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def check_success(self, env) -> bool:
        cube_pos = env.scene[_CUBE_NAME].data.root_pos_w - env.scene.env_origins
        basket_pos = env.scene[_BASKET_NAME].data.root_pos_w - env.scene.env_origins
        dxy = cube_pos[:, :2] - basket_pos[:, :2]
        horizontal_ok = torch.linalg.norm(dxy, dim=-1) < 0.10
        dz = cube_pos[:, 2] - basket_pos[:, 2]
        vertical_ok = (dz > -0.05) & (dz < 0.20)
        return bool((horizontal_ok & vertical_ok).all().item())

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_CUBE_NAME, _BASKET_NAME)
