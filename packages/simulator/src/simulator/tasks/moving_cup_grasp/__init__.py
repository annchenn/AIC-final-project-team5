import gymnasium as gym


gym.register(
    id="HCIS-MovingCubeGrasp-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.moving_cube_grasp_env_cfg:MovingCubeGraspEnvCfg",
    },
)
