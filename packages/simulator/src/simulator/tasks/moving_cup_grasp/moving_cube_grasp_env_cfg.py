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
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.35, 0.0, 0.0)
OBJECT_Z: float = 0.05  # cube half-height above the table surface
CUBE_SIZE: float = 0.05

# Resolve basket USD relative to repo root
#   parents: [0]=moving_cup_grasp [1]=tasks [2]=simulator [3]=src
#   [4]=simulator(pkg) [5]=packages [6]=aicapstone-0531 (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[6]
BASKET_USD_PATH = str(_REPO_ROOT / "basket_23" / "model_basket_23.usd")
BASKET_POS: tuple[float, float, float] = (0.55, -0.45, 0.05)

# Cube workspace bounds (world frame, relative to env origin). The FSM bounces
# the injected velocity off these virtual walls so the cube never slides off
# the table during the chase phase. Tune in tandem with ANCHOR_WORLD_POSE and
# the per-episode object_poses.json initial positions.
CUBE_WORKSPACE_X: tuple[float, float] = (0.10, 0.65)
CUBE_WORKSPACE_Y: tuple[float, float] = (-0.35, 0.30)


@configclass
class MovingCubeGraspSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Living-room scene + a single sliding cube + a basket as drop target."""

    scene: AssetBaseCfg = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/cube",
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

    # Basket as the drop target. Kept rigid (not kinematic) so the FSM can
    # still topple it if mis-placed — mass is high enough that the dropped
    # cube does not knock it around.
    basket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/basket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BASKET_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BASKET_USD_PATH,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    )


def cube_in_basket(
    env,
    cube_cfg: SceneEntityCfg,
    basket_cfg: SceneEntityCfg,
    xy_radius: float,
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Termination: cube xy is within ``xy_radius`` of basket and z within range."""
    cube: RigidObject = env.scene[cube_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    cube_pos = cube.data.root_pos_w - env.scene.env_origins
    basket_pos = basket.data.root_pos_w - env.scene.env_origins

    dxy = cube_pos[:, :2] - basket_pos[:, :2]
    horizontal_ok = torch.linalg.norm(dxy, dim=-1) < xy_radius
    dz = cube_pos[:, 2] - basket_pos[:, 2]
    vertical_ok = (dz > z_range[0]) & (dz < z_range[1])
    return horizontal_ok & vertical_ok


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=cube_in_basket,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "basket_cfg": SceneEntityCfg("basket"),
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
    cube_linear_speed_range: tuple[float, float] = (0.05, 0.10)
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
