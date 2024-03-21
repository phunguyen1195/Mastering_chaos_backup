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


dyn = {"g":9.8, "m2": 1.0, "l": 1.0, "m1": 10.0}
x0 = np.array([0,0,0,0])
v1 = np.array([1, 0, 0, 0], dtype=np.float32)
v2 = np.array([0, 1, 0, 0], dtype=np.float32)
v3 = np.array([0, 0, 1, 0], dtype=np.float32)
v4 = np.array([0, 0, 0, 1], dtype=np.float32)
x_dot = []
x_norm = []
time_step = 1
# cum = np.array([0,0,0])

# def cosin 

def double_pendulum (x0, dyn):
    g = dyn['g']
    l = dyn['l']
    m1 = dyn['m1']
    m2 = dyn['m2']
    #print (x0)
    f = np.array([x0[1],
                ((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 0.5*(x0[3]**2)*np.sin(x0[2]) + np.sin(x0[2])*x0[1]*x0[3] - 4.9*np.cos(x0[0] + x0[2]) - 14.7*np.cos(x0[0])) / (3.5 + np.cos(x0[2])),
                x0[3],
                (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
                ], dtype=np.float32)
    g = np.array([0,
        (-1.25 - 0.5*np.cos(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))),
        0,
        1 / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
        ], dtype=np.float32)
    return f + g*0


def linearized_double_pendulum ( identity, dyn, x0):
    g = dyn['g']
    l = dyn['l']
    m1 = dyn['m1']
    m2 = dyn['m2']
    led_double_pendulum = np.array([[0, 1, 0, 0],
                            [(((-1.25 - 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 14.7*np.sin(x0[0]))) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 4.9*np.sin(x0[0] + x0[2]) + 14.7*np.sin(x0[0])) / (3.5 + np.cos(x0[2])),
                            (((-1.25 - 0.5*np.cos(x0[2]))*((-(1.25 + 0.5*np.cos(x0[2]))*np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])) - np.sin(x0[2])*x0[1])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])),
                            (((-1.25 - 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 0.5*(x0[3]**2)*np.cos(x0[2]) - np.cos(x0[2])*x0[1]*x0[3]) - 0.5*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])*np.sin(x0[2])) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]) + (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) - 0.5*(x0[1]**2)*np.cos(x0[2])) - 0.5*((-(1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) + 4.9*np.cos(x0[0] + x0[2]) + 0.5*(x0[1]**2)*np.sin(x0[2]))*np.sin(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 4.9*np.sin(x0[0] + x0[2]) + 0.5*(x0[3]**2)*np.cos(x0[2]) + np.cos(x0[2])*x0[1]*x0[3] - (((-((1.25 + 0.5*np.cos(x0[2]))**2)) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) + (0.5*(2.5 + np.cos(x0[2]))*np.sin(x0[2])) / (3.5 + np.cos(x0[2])))*((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))**2))) / (3.5 + np.cos(x0[2])) + (((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 0.5*(x0[3]**2)*np.sin(x0[2]) + np.sin(x0[2])*x0[1]*x0[3] - 4.9*np.cos(x0[0] + x0[2]) - 14.7*np.cos(x0[0])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]),
                            (((-1.25 - 0.5*np.cos(x0[2]))*(1.25 + 0.5*np.cos(x0[2]))*(-np.sin(x0[2])*x0[1] - np.sin(x0[2])*x0[3])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))) + np.sin(x0[2])*x0[1] + np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2]))],
                                    [0, 0, 0, 1],
                            [(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 14.7*np.sin(x0[0]))) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))),
                            ((-(1.25 + 0.5*np.cos(x0[2]))*np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])) - np.sin(x0[2])*x0[1]) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))),
                            (((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 0.5*(x0[3]**2)*np.cos(x0[2]) - np.cos(x0[2])*x0[1]*x0[3]) - 0.5*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])*np.sin(x0[2])) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]) + (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) - 0.5*(x0[1]**2)*np.cos(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) - ((((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))**2))*(((-((1.25 + 0.5*np.cos(x0[2]))**2)) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) + (0.5*(2.5 + np.cos(x0[2]))*np.sin(x0[2])) / (3.5 + np.cos(x0[2]))),
                            ((1.25 + 0.5*np.cos(x0[2]))*(-np.sin(x0[2])*x0[1] - np.sin(x0[2])*x0[3])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2])))]
                                ], dtype=np.float32)
    pre_dot = np.dot(led_double_pendulum, identity)
    return pre_dot

def RungeKutta ( dyn, f, dt, x0):

    k1 = f(x0.copy(), dyn) #[x,y,z]*0.1 example
    k2 = f(x0.copy() + 0.5*k1*dt, dyn)
    k3 = f(x0.copy() + 0.5*k2*dt, dyn)
    k4 = f(x0.copy() + k3*dt, dyn)

    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

    return x

def RungeKutta_linearized ( dyn, f, dt, x0, y):
    k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1*dt,dyn, y)
    k3 = f(x0 + 0.5*k2*dt, dyn, y)
    k4 = f(x0 + k3*dt, dyn, y)
    
    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

    return x


def f_t (dyn, f, dt, x0, T):
    x = np.empty(shape=(len(x0),T))
    #print(x.shape)
    x[:, 0] = x0
    # print ('x:', x)
    # print ('x[:, 0]:', x[:, 0])
    # print ('x0:', x0)
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1])
    return x

x = f_t(dyn, double_pendulum, 0.1, x0, time_step)
x[0] = x[0]+ np.pi/2
x = np.array([x[0],x[2],x[1],x[3]])
# print (x)
# input()
env = gym.make("acrobot_le-v0")
# env = gym.make('Pendulum-v1')

new_x = []
#del model # remove to demonstrate saving and loading

model = SAC.load("/home/015970994/masterchaos/models_for_paper/double_pendulum/oct24/sac_double_pendulum_le_300_001_long_0.zip")
env.reset()
# obs = x[:,-1]
# new_x.append(np.array([np.pi-0.5,0,0,0]))
#position, velocity, np.cos(theta), np.sin(theta), thetadot
# obs = np.array([np.cos(obs[0]), np.sin(obs[0]),np.cos(obs[1]), np.sin(obs[1]),obs[2] , obs[3]], dtype=np.float32)
obs = np.array([np.cos(0), np.sin(0),np.cos(0), np.sin(0),0 , 0], dtype=np.float32)

ac = []
for i in range(100):
    # env.render()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    new_x.append(info['trajectory'])
    ac.append(action)
    # env.render()
    # if done:
    #     break
    #   obs = env.reset()


xfinal = np.zeros([4,len(new_x)])
for i in range(len(new_x)):
    xfinal[:,i] = new_x[i]

np.save("double_pendulum_trajectory",xfinal)
# [-8.485, -8.485, 27]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.scatter(x0[0], x0[2], s=50,alpha=1)
ax.plot(x[0], x[2], alpha=0.4)
ax.plot(xfinal[0], xfinal[2], c='green', alpha=0.6)
ax.scatter(xfinal[0][0], xfinal[2][0], s=50, c='green',alpha=1)
ax.scatter(  np.pi,  0, s=100, c='red', marker='o')
# ax.scatter(  0,  0,  0 , s=100, c='red', marker='o')
# [ 0.0070262  -0.03513102  0.03513102]
plt.title('double pendulum with le')
plt.draw()
plt.savefig('double_pendulum_le.png')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(411)
ax.set_title(r"$\mathbf{\Theta_1}$ vs timestep")
ax.set_xlabel('timestep')
ax.set_ylabel(r"$\mathbf{\Theta_1}$")
# ax.plot(np.array(list(range(len(x[0])))), x[0])
ax.scatter(np.array(list(range(len(xfinal[0])))), xfinal[0], c = 'green')
ax.axhline(y = np.pi, color = 'r', linestyle = 'dashdot')



ax1 = fig.add_subplot(412)
ax1.set_title(r"$\mathbf{\Theta_2}$ vs timestep")
ax1.set_xlabel('timestep')
ax1.set_ylabel(r"$\mathbf{\Theta_2}$")
# ax1.plot(np.array(list(range(len(x[1])))), x[1])
ax1.scatter(np.array(list(range(len(xfinal[1])))), xfinal[1], c = 'green')
ax1.axhline(y = 0, color = 'r', linestyle = 'dashdot')

ax1 = fig.add_subplot(413)
ax1.set_title(r"$\mathbf{\dot{\Theta_1}}$ vs timestep")
ax1.set_xlabel('timestep')
ax1.set_ylabel(r"$\mathbf{\dot{\Theta_1}}$")
# ax1.plot(np.array(list(range(len(x[2])))), x[2])
ax1.scatter(np.array(list(range(len(xfinal[2])))), xfinal[2], c = 'green')
ax1.axhline(y = 0, color = 'r', linestyle = 'dashdot')

ax1 = fig.add_subplot(414)
ax1.set_title(r"$\mathbf{\dot{\Theta_2}}$ vs timestep")
ax1.set_xlabel('timestep')
ax1.set_ylabel(r"$\mathbf{\dot{\Theta_2}}$")
# ax1.plot(np.array(list(range(len(x[3])))), x[3])
ax1.scatter(np.array(list(range(len(xfinal[3])))), xfinal[3], c = 'green')
ax1.axhline(y = 0, color = 'r', linestyle = 'dashdot')

fig.suptitle('Trajectory Stabilization', fontsize=18)
fig.tight_layout()
plt.savefig('coordinatesvstime_double_pendulum_le.png')

# env.reset()
# print(env.step(env.action_space.sample()))