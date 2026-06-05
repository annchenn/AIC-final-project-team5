"""Advanced-level task: grasp a sliding cube and drop it into a basket.

Builds directly on the Entry-level ``SingleArmFrankaTaskSceneCfg`` template.
A single rigid cube is spawned in the living-room scene; the state machine
in ``simulator.datagen.state_machine.moving_cube_grasp`` injects a constant
linear velocity each tick (until grasp), so the policy must learn to chase,
intercept, lift, and place the moving target into the basket.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
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
