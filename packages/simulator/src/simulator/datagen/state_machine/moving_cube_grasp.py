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
# 60 Hz step rate. Velocity is injected only through phase 1 — from phase 2
# the cube decelerates from table friction and stops within ~25 ms, so the
# gripper closes on a near-stationary target. This is the same pattern as the
# original (working) commit 3b7e97d, just extended with basket placement.
_PHASE_DURATIONS = (120, 180, 40, 80, 140, 30, 40)
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
        self._dump_table_bbox(env)
        self._sample_cube_velocity(env)

    def _dump_table_bbox(self, env) -> None:
        """Print table-top world AABB once, so we can tune workspace bounds."""
        try:
            import isaacsim.core.utils.bounds as bounds_utils
            import isaacsim.core.utils.prims as prims_utils

            # Living-room scene is mounted under each env at Scene/...; the
            # actual table prim's name comes from the USD (e.g. 'table' or
            # 'counter_right_main_group'). Try a few.
            origins = env.scene.env_origins[0]
            env_ns = "/World/envs/env_0/Scene"
            candidates = [
                f"{env_ns}/counter_right_main_group",
                f"{env_ns}/table",
                f"{env_ns}/Table",
                f"{env_ns}/desk",
            ]
            cache = bounds_utils.create_bbox_cache()
            for path in candidates:
                if not prims_utils.is_prim_path_valid(path):
                    continue
                aabb = bounds_utils.compute_aabb(cache, path)  # (xmin,ymin,zmin,xmax,ymax,zmax)
                local = (
                    aabb[0] - float(origins[0]), aabb[1] - float(origins[1]), aabb[2] - float(origins[2]),
                    aabb[3] - float(origins[0]), aabb[4] - float(origins[1]), aabb[5] - float(origins[2]),
                )
                print(
                    f"[table-bbox] prim={path}\n"
                    f"             world AABB min=({aabb[0]:.3f},{aabb[1]:.3f},{aabb[2]:.3f}) "
                    f"max=({aabb[3]:.3f},{aabb[4]:.3f},{aabb[5]:.3f})\n"
                    f"             local AABB (env-frame) "
                    f"min=({local[0]:.3f},{local[1]:.3f},{local[2]:.3f}) "
                    f"max=({local[3]:.3f},{local[4]:.3f},{local[5]:.3f})"
                )
        except Exception as e:
            print(f"[table-bbox] dump failed: {e}")

    def reset(self) -> None:
        super().reset()
        self._cube_velocity_w = None  # resampled on next pre_step

    def _sample_cube_velocity(self, env) -> None:
        """Rejection-sample a velocity that keeps the cube on the table.

        Direction is biased toward the workspace centre (so the cube heads
        inward, never outward); rejection sampling on the projected endpoint
        is the safety net.
        """
        device = env.device
        num_envs = env.num_envs
        cube_pos = env.scene[_CUBE_NAME].data.root_pos_w - env.scene.env_origins
        x0 = cube_pos[:, 0]
        y0 = cube_pos[:, 1]

        dt = env.physics_dt * env.cfg.decimation
        chase_time = (self._events_dt[0] + self._events_dt[1]) * dt + 0.4

        x_min, x_max = self._workspace_x
        y_min, y_max = self._workspace_y
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        margin = 0.05
        x_lo, x_hi = x_min + margin, x_max - margin
        y_lo, y_hi = y_min + margin, y_max - margin

        # Bearing from cube to workspace centre — primary direction bias.
        bearing = torch.atan2(cy - y0, cx - x0)

        vx = torch.zeros(num_envs, device=device)
        vy = torch.zeros(num_envs, device=device)
        accepted = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for _ in range(50):
            speed = torch.empty(num_envs, device=device).uniform_(*self._speed_range)
            jitter = torch.empty(num_envs, device=device).uniform_(
                -math.pi / 3.0, math.pi / 3.0
            )
            angle = bearing + jitter
            vx_c = speed * torch.cos(angle)
            vy_c = speed * torch.sin(angle)
            xe = x0 + vx_c * chase_time
            ye = y0 + vy_c * chase_time
            inside = (xe > x_lo) & (xe < x_hi) & (ye > y_lo) & (ye < y_hi)
            take = (~accepted) & inside
            vx = torch.where(take, vx_c, vx)
            vy = torch.where(take, vy_c, vy)
            accepted = accepted | take
            if bool(accepted.all().item()):
                break

        if not bool(accepted.all().item()):
            # Fallback: aim straight at the workspace centre at slowest speed.
            slow = torch.full((num_envs,), self._speed_range[0], device=device)
            vx_fb = slow * torch.cos(bearing)
            vy_fb = slow * torch.sin(bearing)
            vx = torch.where(~accepted, vx_fb, vx)
            vy = torch.where(~accepted, vy_fb, vy)

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

        # Stop driving the cube once we start closing the gripper. Phase 2+
        # lets table friction decelerate the cube within ~25 ms so the gripper
        # closes on a near-stationary target.
        if self._event >= 2:
            return

        # Wait for the cube to settle on the table before injecting velocity.
        # We spawn slightly above the surface; the first few ticks are free
        # fall, during which we must NOT override vz or the cube hovers.
        if self._event == 0 and self._step_count < 20:
            return

        cube = env.scene[_CUBE_NAME]
        # Read current vz so gravity / table contact are preserved; only
        # overwrite the horizontal components (the constant we want).
        current_vel = cube.data.root_vel_w.clone()
        vel = self._cube_velocity_w.to(device=env.device, dtype=current_vel.dtype).clone()
        vel[:, 2] = current_vel[:, 2]      # keep vz from physics
        vel[:, 3:] = current_vel[:, 3:]    # keep angular velocity from physics
        cube.write_root_velocity_to_sim(vel)

        # One-time sanity print: confirm cube actually moves.
        if self._event == 1 and self._step_count == 0:
            cp = cube.data.root_pos_w[0] - env.scene.env_origins[0]
            cv = current_vel[0]
            print(
                f"[moving_cube] phase-1 start cube pos=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) "
                f"actual vel=({cv[0]:+.3f},{cv[1]:+.3f},{cv[2]:+.3f})"
            )
        if self._event == 1 and self._step_count == self._events_dt[1] - 1:
            cp = cube.data.root_pos_w[0] - env.scene.env_origins[0]
            cv = cube.data.root_vel_w[0]
            print(
                f"[moving_cube] phase-1 end   cube pos=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) "
                f"actual vel=({cv[0]:+.3f},{cv[1]:+.3f},{cv[2]:+.3f})"
            )

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

        # Basket is a static AssetBase, not a RigidObject — read pose from cfg.
        basket_xyz = torch.tensor(env.cfg.basket_pos, device=device, dtype=cube_pos_w.dtype)
        basket_pos_w = basket_xyz.unsqueeze(0).expand(num_envs, 3) + env.scene.env_origins

        if self._step_count == 0 and self._event == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        target_quat_w = self._gripper_down_quat_w(
            cube_quat_w, num_envs, device, cube_quat_w.dtype, yaw_offset=_GRASP_YAW_OFFSET
        )

        # Lead the cube by exactly the remaining steps in phase 1, matching
        # the original working FSM. At the END of phase 1, lead == 0, so the
        # EE is on top of the cube's current position. Phase 2 then stops
        # injecting velocity (see pre_step), the cube decelerates from
        # friction, and the gripper closes on a near-stationary target.
        dt = env.physics_dt * env.cfg.decimation
        if self._cube_velocity_w is not None:
            vel_xy = self._cube_velocity_w[:, :2].to(target_quat_w.device, dtype=cube_pos_w.dtype)
        else:
            vel_xy = None

        phase = self._event
        if phase == 0:
            # Hover above the (moving) cube; smoothly interpolate from rest.
            # No lead here — phase 0 just transports the EE to hover height
            # over the cube's current XY; phase 1 takes over with predictive
            # lead based on its own remaining time.
            target = cube_pos_w.clone()
            target[:, 2] += _HOVER_Z_OFFSET
            if self._initial_ee_pos_w is not None:
                denom = max(self._events_dt[self._event] - 1, 1)
                alpha = min(self._step_count / denom, 1.0)
                target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
            gripper = _constant_gripper(num_envs, device, _GRIPPER_OPEN)
        elif phase == 1:
            # Approach: compute moving target that tracks cube XY and interpolates Z
            target = cube_pos_w.clone()
            
            # Smoothly descend from hover height to grasp height 
            # We add an extra 0.1m to _GRASP_Z_OFFSET so we don't smash the table
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target[:, 2] += (1.0 - alpha) * _HOVER_Z_OFFSET + alpha * (_GRASP_Z_OFFSET + 0.1)
            
            gripper = _constant_gripper(num_envs, device, _GRIPPER_OPEN)
        elif phase == 2:
            # Close gripper at cube's current position. Velocity injection has
            # stopped (pre_step) so the cube decelerates and the fingers can
            # close around it.
            target = cube_pos_w.clone()
            target[:, 2] += _GRASP_Z_OFFSET + 0.1
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
            # Lower toward the basket interior. Hold the cube just above the
            # basket walls (~10 cm above basket floor); phase 6 opens the
            # gripper and the cube drops the rest of the way in.
            target = basket_pos_w.clone()
            target[:, 2] += 0.10
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
        basket_pos = torch.tensor(env.cfg.basket_pos, device=cube_pos.device, dtype=cube_pos.dtype)
        dxy = cube_pos[:, :2] - basket_pos[:2]
        horizontal_ok = torch.linalg.norm(dxy, dim=-1) < 0.10
        dz = cube_pos[:, 2] - basket_pos[2]
        vertical_ok = (dz > -0.05) & (dz < 0.20)
        return bool((horizontal_ok & vertical_ok).all().item())

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_CUBE_NAME,)
