from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

_load_env_plugins()
# try 2000
register(id='lorenz_u-v0',entry_point='Lorenz_envs.envs:Lorenzu', max_episode_steps=1000,) 

register(id='rossler-v0',entry_point='Lorenz_envs.envs:Rossler', max_episode_steps=6500,)

register(id='lorenz_y-v0',entry_point='Lorenz_envs.envs:Lorenzy', max_episode_steps=1000,) 

register(id='lorenz_le-v0',entry_point='Lorenz_envs.envs:Lorenzle', max_episode_steps=1000,)

register(id='pendulum_le-v0',entry_point='Lorenz_envs.envs:pendulum_le', max_episode_steps=200,)

register(id='pendulum_he-v0',entry_point='Lorenz_envs.envs:pendulum_he', max_episode_steps=200,)

register(id='cartpole_le-v0',entry_point='Lorenz_envs.envs:cartpole_le', max_episode_steps=500,)

register(id='double_pendulum_le-v0',entry_point='Lorenz_envs.envs:double_pendulum_le', max_episode_steps=500,)

register(id='acrobot_le-v0',entry_point='Lorenz_envs.envs:AcrobotEnv', max_episode_steps=500,)

register(id='acrobot_le-v1',entry_point='Lorenz_envs.envs:AcrobotEnv_origin', max_episode_steps=500,)