"""Advanced-level task: grasp a cube that is sliding across the table.

Builds directly on the Entry-level ``SingleArmFrankaTaskSceneCfg`` template.
A single rigid cube is spawned in the living-room scene; the state machine
in ``simulator.datagen.state_machine.moving_cube_grasp`` injects a constant
linear velocity each tick (until grasp), so the policy must learn to chase
and intercept a moving target.
"""

from __future__ import annotations

import math

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


@configclass
class MovingCubeGraspSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Living-room scene + a single sliding cube."""

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


def cube_lifted(
    env,
    cube_cfg: SceneEntityCfg,
    lift_height: float,
) -> torch.Tensor:
    """Termination: cube has been lifted ``lift_height`` metres above its rest z."""
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos = cube.data.root_pos_w - env.scene.env_origins
    return cube_pos[:, 2] > (OBJECT_Z + lift_height)


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=cube_lifted,
        params={"cube_cfg": SceneEntityCfg("cube"), "lift_height": 0.12},
    )


@configclass
class MovingCubeGraspEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: MovingCubeGraspSceneCfg = MovingCubeGraspSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "grasp the cube that is sliding across the table and lift it."

    # State-machine consumed knobs (also read by the FSM via env_cfg).
    cube_linear_speed_range: tuple[float, float] = (0.05, 0.10)
    cube_lift_threshold: float = 0.12

    def __post_init__(self) -> None:
        super().__post_init__()
        # Longer than entry-level tasks: chasing + grasping takes more steps.
        self.episode_length_s = 20

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
