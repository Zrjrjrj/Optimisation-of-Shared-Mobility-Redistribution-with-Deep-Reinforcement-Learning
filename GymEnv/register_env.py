from gym.envs.registration import register

register(
    id='RedistributionEnv-v0',
    entry_point='GymEnv.redistribution_env:RedistributionEnv2',
)
