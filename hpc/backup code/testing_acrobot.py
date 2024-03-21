import gym
import numpy as np
import Lorenz_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EventCallback
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3.common.logger import Figure



import numpy as np
from fractions import Fraction
directory = 'lorenz/'



env = gym.make("Acrobot-v1")


model = PPO('MlpPolicy', env ,gamma=0.985, 
    # use_sde=True,
    # sde_sample_freq=4,
    # learning_rate=linear_schedule(1e-3),
    # learning_rate=linear_schedule(4e-3),
    learning_rate=1e-3, 
    tensorboard_log="./double_pendulum_tensorboard/")
model.learn(total_timesteps=100000)

model.save("models_for_paper/double_pendulum/ppo_double_pendulum_le_2")
