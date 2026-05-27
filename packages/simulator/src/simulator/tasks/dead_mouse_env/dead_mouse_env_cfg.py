import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
)


@configclass
class DeadMouseEnvSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the dead mouse environment."""

    red_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="${ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.4, 0.12),
            rot=(1, 0, 0, 0),
        ),
    )

    blue_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="${ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.6, 0.12),
            rot=(1, 0, 0, 0),
        ),
    )

    green_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.8, 0.12),
            rot=(1, 0, 0, 0),
        ),
    )


@configclass
class DeadMouseEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the dead mouse environment."""

    scene: DeadMouseEnvSceneCfg = DeadMouseEnvSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    task_description: str = "pick up the blocks and place them in the target location."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (0.5, -0.6, 0.0)

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
