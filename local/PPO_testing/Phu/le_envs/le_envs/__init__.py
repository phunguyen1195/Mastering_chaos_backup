from gymnasium.envs.registration import register

register(
    id="le_envs/Pendulum_le-v0",
    entry_point="le_envs.envs:PendulumLeEnv",
    vector_entry_point="le_envs.envs:PendulumLeVectorEnv",
    max_episode_steps=200
)

register(
    id="le_envs/Lorenz-v0",
    entry_point="le_envs.envs:LorenzEnv",
    vector_entry_point="le_envs.envs:LorenzVectorEnv",
    max_episode_steps=500
)