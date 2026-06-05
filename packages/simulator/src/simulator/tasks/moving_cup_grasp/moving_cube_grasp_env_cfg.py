"""Advanced-level task: grasp a sliding cube and drop it into a basket.

Builds directly on the Entry-level ``SingleArmFrankaTaskSceneCfg`` template.
A single rigid cube is spawned in the living-room scene; the state machine
in ``simulator.datagen.state_machine.moving_cube_grasp`` injects a constant
linear velocity each tick (until grasp), so the policy must learn to chase,
intercept, lift, and place the moving target into the basket.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator.assets.scenes.living_room import LIVING_ROOM_CFG, LIVING_ROOM_USD_PATH
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)
from simulator.utils.object_poses_loader import ObjectPoseConfig

# Tag → object mapping for the UMI episode-poses JSON. Only one cube here.
TAG_TO_OBJECT: dict[int, str] = {1: "cube"}
ANCHOR_TAG_ID: int = 0
CUBE_SIZE: float = 0.05
# Anchor at the centre of the measured table AABB (x ∈ [0.003, 0.703],
# y ∈ [-0.677, -0.027], z_top = 0.041). Per-episode tvec is added in this
# anchor frame, so the JSON values are simple table-local offsets.
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.353, -0.352, 0.0)
# Cube half-height (0.025) just above the measured table-top z (0.041). A
# 2 mm clearance avoids spawn-time interpenetration with the table mesh.
OBJECT_Z: float = 0.041 + CUBE_SIZE * 0.5 + 0.002  # = 0.068

# Resolve basket USD relative to repo root
#   parents: [0]=moving_cup_grasp [1]=tasks [2]=simulator [3]=src
#   [4]=simulator(pkg) [5]=packages [6]=aicapstone-0531 (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[6]
BASKET_USD_PATH = str(_REPO_ROOT / "basket_23" / "model_basket_23.usd")
# Basket placed near the +x, -y corner of the table (within measured AABB,
# leaving room for the basket footprint and the cube). Table top is z=0.041,
# so basket bottom sits exactly on it.
BASKET_POS: tuple[float, float, float] = (0.58, -0.55, 0.041)

# Cube workspace bounds (world frame, env-local) — matches the MEASURED table
# AABB: x ∈ [0.003, 0.703], y ∈ [-0.677, -0.027]. The FSM's rejection sampler
# uses these to guarantee the cube cannot slide off the table.
CUBE_WORKSPACE_X: tuple[float, float] = (0.05, 0.65)
CUBE_WORKSPACE_Y: tuple[float, float] = (-0.65, -0.05)


# ---------------------------------------------------------------------------
# Drop-target basket geometry (built from 5 collision cuboids: bottom + 4 walls).
# The basket_23 USD ships with rigid-body APIs on multiple sub-prims and makes
# physx-fabric crash on GPU when loaded as either RigidObject or AssetBase, so
# we construct a real basket out of primitives instead. Each piece is a static
# AssetBase with collision enabled — the cube can land in it and rest, and the
# success-termination z-range is satisfied.
# ---------------------------------------------------------------------------
_BASKET_INNER: float = 0.18    # inner footprint side (m)
_BASKET_WALL: float = 0.01     # wall thickness (m)
_BASKET_HEIGHT: float = 0.08   # wall height (m)
_BASKET_BOTTOM_THK: float = 0.01
_BASKET_OUTER: float = _BASKET_INNER + 2.0 * _BASKET_WALL
_BX, _BY, _BZ = BASKET_POS


def _basket_piece_cfg(
    name: str, offset: tuple[float, float, float], size: tuple[float, float, float],
    color: tuple[float, float, float] = (0.9, 0.6, 0.2),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/drop_basket_" + name,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(_BX + offset[0], _BY + offset[1], _BZ + offset[2]),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.0, roughness=0.8
            ),
        ),
    )


@configclass
class MovingCubeGraspSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Living-room scene + a sliding cube + a drop basket built from 5 cuboids."""

    scene: AssetBaseCfg = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Default spawn at workspace centre on the table top.
            # During datagen this is overridden by object_poses.json; during
            # rollout there is no --object_poses arg so this value is used.
            pos=(0.35, -0.35, OBJECT_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.3, 0.9), metallic=0.1, roughness=0.6
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.2,
                restitution=0.0,
            ),
        ),
    )

    # Basket: bottom plate at z=0 (basket-local), 4 walls around it.
    basket_bottom: AssetBaseCfg = _basket_piece_cfg(
        "bottom",
        offset=(0.0, 0.0, _BASKET_BOTTOM_THK * 0.5),
        size=(_BASKET_OUTER, _BASKET_OUTER, _BASKET_BOTTOM_THK),
    )
    basket_wall_xp: AssetBaseCfg = _basket_piece_cfg(
        "wall_xp",
        offset=(+(_BASKET_INNER + _BASKET_WALL) * 0.5, 0.0, _BASKET_HEIGHT * 0.5),
        size=(_BASKET_WALL, _BASKET_OUTER, _BASKET_HEIGHT),
    )
    basket_wall_xn: AssetBaseCfg = _basket_piece_cfg(
        "wall_xn",
        offset=(-(_BASKET_INNER + _BASKET_WALL) * 0.5, 0.0, _BASKET_HEIGHT * 0.5),
        size=(_BASKET_WALL, _BASKET_OUTER, _BASKET_HEIGHT),
    )
    basket_wall_yp: AssetBaseCfg = _basket_piece_cfg(
        "wall_yp",
        offset=(0.0, +(_BASKET_INNER + _BASKET_WALL) * 0.5, _BASKET_HEIGHT * 0.5),
        size=(_BASKET_INNER, _BASKET_WALL, _BASKET_HEIGHT),
    )
    basket_wall_yn: AssetBaseCfg = _basket_piece_cfg(
        "wall_yn",
        offset=(0.0, -(_BASKET_INNER + _BASKET_WALL) * 0.5, _BASKET_HEIGHT * 0.5),
        size=(_BASKET_INNER, _BASKET_WALL, _BASKET_HEIGHT),
    )




def cube_in_basket(
    env,
    cube_cfg: SceneEntityCfg,
    xy_radius: float,
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Termination: cube xy is within ``xy_radius`` of basket and z within range."""
    cube: RigidObject = env.scene[cube_cfg.name]

    cube_pos = cube.data.root_pos_w - env.scene.env_origins
    basket_pos = torch.tensor(env.cfg.basket_pos, device=cube_pos.device, dtype=cube_pos.dtype)

    dxy = cube_pos[:, :2] - basket_pos[:2]
    horizontal_ok = torch.linalg.norm(dxy, dim=-1) < xy_radius
    dz = cube_pos[:, 2] - basket_pos[2]
    vertical_ok = (dz > z_range[0]) & (dz < z_range[1])
    return horizontal_ok & vertical_ok


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=cube_in_basket,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "xy_radius": 0.10,
            "z_range": (-0.05, 0.20),
        },
    )


@configclass
class MovingCubeGraspEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: MovingCubeGraspSceneCfg = MovingCubeGraspSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "grasp the sliding cube and drop it into the basket."

    # State-machine consumed knobs (also read by the FSM via env_cfg).
    cube_linear_speed_range: tuple[float, float] = (0.10, 0.18)
    cube_lift_threshold: float = 0.12
    cube_workspace_x: tuple[float, float] = CUBE_WORKSPACE_X
    cube_workspace_y: tuple[float, float] = CUBE_WORKSPACE_Y
    basket_pos: tuple[float, float, float] = BASKET_POS

    def __post_init__(self) -> None:
        super().__post_init__()
        # Longer than entry-level tasks: chasing + grasping + placing.
        self.episode_length_s = 30

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(LIVING_ROOM_USD_PATH, self)

        self.object_pose_cfg = ObjectPoseConfig(
            tag_to_object=TAG_TO_OBJECT,
            anchor_tag_id=ANCHOR_TAG_ID,
            anchor_world_pose=ANCHOR_WORLD_POSE,
            object_z=OBJECT_Z,
            object_roll=0.0,
            object_pitch=0.0,
            per_object_yaw_offset={"cube": 0.0},
            use_fixed_yaw=True,
        )


class MovingCubeGraspEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that injects constant cube velocity every step.

    Works in both datagen and rollout — no FSM required at inference time.
    """

    # Number of physics steps to wait after reset before injecting velocity.
    # Matches FSM pre_step: cube spawns 2 mm above table and needs a few ticks
    # to land; injecting vz-override too early makes the cube hover.
    _SETTLE_STEPS: int = 20

    def __init__(self, cfg: MovingCubeGraspEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cube_vel_w: torch.Tensor | None = None
        self._needs_resample: torch.Tensor | None = None
        self._steps_since_reset: torch.Tensor | None = None
        self._json_positions: list[tuple[float, float]] = self._load_json_positions()

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        if self._needs_resample is None:
            self._needs_resample = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self._needs_resample[env_ids] = True
            self._steps_since_reset[env_ids] = 0
        self._randomise_cube_position(env_ids)

    @staticmethod
    def _load_json_positions() -> list[tuple[float, float]]:
        """Load cube (x, y) world positions from object_poses.json.

        Falls back to empty list if the file is missing; _randomise_cube_position
        will then fall back to uniform random.
        """
        json_path = _REPO_ROOT / "data" / "moving_cube_demo" / "object_poses.json"
        if not json_path.is_file():
            print(f"[MovingCubeGraspEnv] object_poses.json not found at {json_path}, "
                  "falling back to uniform random spawn.", flush=True)
            return []
        with json_path.open() as f:
            entries = json.load(f)
        ax, ay = ANCHOR_WORLD_POSE[0], ANCHOR_WORLD_POSE[1]  # anchor_yaw = 0
        positions = []
        for entry in entries:
            if entry.get("status") != "full":
                continue
            for obj in entry.get("objects", []):
                if obj.get("object_name") == "cube":
                    tvec = obj["tvec"]
                    positions.append((ax + tvec[0], ay + tvec[1]))
        print(f"[MovingCubeGraspEnv] loaded {len(positions)} cube positions from object_poses.json",
              flush=True)
        return positions

    def _randomise_cube_position(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        device = self.device
        if self._json_positions:
            indices = torch.randint(len(self._json_positions), (n,))
            pos_list = [self._json_positions[i] for i in indices.tolist()]
            rx = torch.tensor([p[0] for p in pos_list], device=device)
            ry = torch.tensor([p[1] for p in pos_list], device=device)
        else:
            x_min, x_max = self.cfg.cube_workspace_x
            y_min, y_max = self.cfg.cube_workspace_y
            margin = 0.08
            rx = torch.empty(n, device=device).uniform_(x_min + margin, x_max - margin)
            ry = torch.empty(n, device=device).uniform_(y_min + margin, y_max - margin)
        cube = self.scene["cube"]
        root_state = cube.data.root_state_w[env_ids].clone()
        root_state[:, 0] = self.scene.env_origins[env_ids, 0] + rx
        root_state[:, 1] = self.scene.env_origins[env_ids, 1] + ry
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + OBJECT_Z
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        root_state[:, 7:] = 0.0
        cube.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _sample_cube_velocity(self, env_ids: torch.Tensor) -> None:
        device = self.device
        n = len(env_ids)
        speed_lo, speed_hi = self.cfg.cube_linear_speed_range
        x_min, x_max = self.cfg.cube_workspace_x
        y_min, y_max = self.cfg.cube_workspace_y
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        margin = 0.05
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        x0, y0 = cube_pos[:, 0], cube_pos[:, 1]
        bearing = torch.atan2(cy - y0, cx - x0)
        if self._cube_vel_w is None:
            self._cube_vel_w = torch.zeros(self.num_envs, 6, device=device)
        dt = self.physics_dt * self.cfg.decimation
        chase_time = (120 + 180) * dt + 0.4
        x_lo, x_hi = x_min + margin, x_max - margin
        y_lo, y_hi = y_min + margin, y_max - margin
        vx = torch.zeros(n, device=device)
        vy = torch.zeros(n, device=device)
        accepted = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(50):
            speed = torch.empty(n, device=device).uniform_(speed_lo, speed_hi)
            jitter = torch.empty(n, device=device).uniform_(-math.pi / 3, math.pi / 3)
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
            if bool(accepted.all()):
                break
        slow = torch.full((n,), speed_lo, device=device)
        vx = torch.where(~accepted, slow * torch.cos(bearing), vx)
        vy = torch.where(~accepted, slow * torch.sin(bearing), vy)
        self._cube_vel_w[env_ids, 0] = vx
        self._cube_vel_w[env_ids, 1] = vy
        self._cube_vel_w[env_ids, 2:] = 0.0
        v0 = self._cube_vel_w[env_ids[0]]
        print(
            f"[MovingCubeGraspEnv] cube velocity sampled:"
            f" vx={v0[0]:+.3f} vy={v0[1]:+.3f} m/s",
            flush=True,
        )

    def _inject_cube_velocity(self) -> None:
        """Write stored cube velocity to sim for envs where cube is still on the table.

        Mirrors FSM pre_step: skip the first _SETTLE_STEPS steps so the cube
        lands on the table before we override its horizontal velocity.
        """
        if self._cube_vel_w is None or self._steps_since_reset is None:
            return
        cube = self.scene["cube"]
        lift_threshold: float = self.cfg.cube_lift_threshold
        cube_z_local = cube.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        settled = self._steps_since_reset >= self._SETTLE_STEPS
        still_on_table = cube_z_local < lift_threshold
        active = (settled & still_on_table).nonzero(as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            return
        current_vel = cube.data.root_vel_w.clone()
        vel = self._cube_vel_w.to(dtype=current_vel.dtype).clone()
        vel[:, 2] = current_vel[:, 2]
        vel[:, 3:] = current_vel[:, 3:]
        cube.write_root_velocity_to_sim(vel[active], env_ids=active)

    def step(self, action: torch.Tensor):
        if self._needs_resample is not None and self._needs_resample.any():
            resample_ids = self._needs_resample.nonzero(as_tuple=False).squeeze(-1)
            self._sample_cube_velocity(resample_ids)
            self._needs_resample[resample_ids] = False
        self._inject_cube_velocity()
        result = super().step(action)
        if self._steps_since_reset is not None:
            self._steps_since_reset += 1
        return result
