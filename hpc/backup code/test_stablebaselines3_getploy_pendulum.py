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


dyn_pendulum = {"g":9.81, "m": 1.0, "l": 1.0}
x0 = np.array([np.pi/4, 0.0])
time_step = 100
def simple_pendulum (x0, dyn):
    g = dyn['g'] 
    l = dyn['l']
    #print (x0)
    return np.array([x0[1], (-g/l)*np.sin(x0[0])])

def RungeKutta (dyn, f, dt, x0):
    k1 = f(x0, dyn) #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1*dt,dyn)
    k3 = f(x0 + 0.5*k2*dt, dyn)
    k4 = f(x0 + k3*dt, dyn)
    
    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) *dt
    return x

def f_t (dyn, f, dt, x0, T):
    x = np.empty(shape=(len(x0),T))
    x[:, 0] = x0     
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1]) 
    return x

x = f_t(dyn_pendulum, simple_pendulum, 0.1, x0, time_step)

env = gym.make("pendulum_le-v0")
# env = gym.make('Pendulum-v1')

new_x = []
#del model # remove to demonstrate saving and loading

model = SAC.load("/home/015970994/masterchaos/models_for_paper/Pendulum/oct27/sac_pendulum_le_final.zip")
env.reset()
obs = x[:,-1]
# new_x.append(np.array([0,0]))
obs = np.array([np.cos(0), np.sin(0), 0], dtype=np.float32)
ac = []
for i in range(200):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    new_x.append(info['velocity'])
    ac.append(action)
    # env.render()
    if done:
        break
    #   obs = env.reset()


xfinal = np.zeros([2,len(new_x)])
for i in range(len(new_x)):
    xfinal[:,i] = new_x[i]

# [-8.485, -8.485, 27]
np.save("pendulum_trajectory",xfinal)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.scatter(x0[0], x0[1], s=50,alpha=1)
ax.plot(x[0], x[1], alpha=0.4)
ax.plot(xfinal[0], xfinal[1], c='green', alpha=0.6)
ax.scatter(xfinal[0][0], xfinal[1][0], s=50, c='green',alpha=1)
ax.scatter(  np.pi,  0, s=100, c='red', marker='o')
# ax.scatter(  0,  0,  0 , s=100, c='red', marker='o')
# [ 0.0070262  -0.03513102  0.03513102]
plt.title('pendulum with le')
plt.draw()
plt.savefig('pendulum_le.png')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.set_title("action vs. timestep")
ax.set_xlabel('timestep')
ax.set_ylabel("a")
# ax.plot(np.array(list(range(len(x[0])))), x[0])
ax.plot(np.array(list(range(len(ac)))), ac, c = 'green')
# ax.axhline(y = np.pi, color = 'r', linestyle = 'dashdot')
fig.suptitle('Simple Pendulum action Stablization', fontsize=18)
fig.tight_layout()
plt.savefig('coordinatesvstime_pendulum_le.png')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.set_title(r"$\mathbf{\Theta_1}$ vs. timestep")
ax.set_xlabel('timestep')
ax.set_ylabel(r"$\mathbf{\Theta_1}$")
# ax.plot(np.array(list(range(len(x[0])))), x[0])
ax.plot(np.array(list(range(len(xfinal[0])))), xfinal[0], c = 'green')
ax.axhline(y = -np.pi, color = 'r', linestyle = 'dashdot')
ax.axhline(y = np.pi, color = 'r', linestyle = 'dashdot')



ax1 = fig.add_subplot(212)
ax1.set_title(r"$\mathbf{\dot{\Theta_1}}$ vs. timestep")
ax1.set_xlabel('timestep')
ax1.set_ylabel(r"$\mathbf{\dot{\Theta_1}}$")
# ax1.plot(np.array(list(range(len(x[1])))), x[1])
ax1.plot(np.array(list(range(len(xfinal[1])))), xfinal[1], c = 'green')
ax1.axhline(y = 0, color = 'r', linestyle = 'dashdot')


fig.suptitle('Simple Pendulum Stablization', fontsize=18)
fig.tight_layout()
plt.savefig('coordinatesvstime_pendulum_le.png', transparent=True)

# env.reset()
# print(env.step(env.action_space.sample()))