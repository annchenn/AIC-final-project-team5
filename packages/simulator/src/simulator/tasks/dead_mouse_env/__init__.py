import gymnasium as gym

gym.register(
    id="LeIsaac-HCIS-DeadMouseEnv-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dead_mouse_env_cfg:DeadMouseEnvCfg",
    },
)
