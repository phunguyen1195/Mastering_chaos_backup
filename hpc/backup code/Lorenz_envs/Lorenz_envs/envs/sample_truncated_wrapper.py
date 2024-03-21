from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
#from gym.utils.renderer import Renderer
from gym.error import DependencyNotInstalled
#from gym.utils.renderer import Renderer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

class Sample_le_wrapper(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        super(Sample_le_wrapper, self).__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")
        self.close_to_goal = False
        self.infinite = False
        self.goal_state = np.array([np.pi, 0.0])
        self.max_speed = 8
        self.max_torque = 2.0
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.1
        self.T = 500
        self.dyn = {"g":self.g , "m":self.m, "l": self.l}
        self.alpha = 3
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.high_env_range = np.array([2*np.pi, self.max_speed], dtype=np.float32)
        self.low_env_range = np.array([0, -self.max_speed], dtype=np.float32)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape = (1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape = (3,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def simple_pendulum (self, x0, dyn, action):
        g = dyn['g'] 
        l = dyn['l']
        m = dyn['m']
        #print (x0)
        # print (action)
        return np.array([x0[1] , (-g/l)*np.sin(x0[0]) + self.alpha * (1.0 / m) * action[0]])


    def RungeKutta (self, dyn, f, dt, x0, action):

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x, k4

    
    def f_t (self, dyn, f, dt, x0, T):
        x = np.empty(shape=(len(x0),T))
        x[:, 0] = x0    
        for i in range(1,T):
            x[:, i] = self.RungeKutta(dyn, f, dt, x[:, i-1]) 
        return x
    
    def get_sample (self, dyn, f, dt, x0, length):
        x, _ = self.f_t (self, dyn, f, dt, x0, length)
        return x
    
    def sample_truncated_le (self, ss):
        pass
    
    def step(self, u):
        x = self.state  # th := theta

        T = self.T
        dyn = self.dyn
        f = self.simple_pendulum
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)
        newx = self.get_sample ( dyn, f, dt, x, T)

        newx[1] = np.clip(newx[1], self.low_env_range[1], self.high_env_range[1])

        self.cost = self.sample_truncated_le(newx)

        terminated = False

        self.state = newx

        return self._get_obs(), self.cost, terminated, {'velocity': x, 'action': u}

    def reset(self):
        #put even closer to reward
        # high = np.array([np.pi, 1.0])
        # self.state = self.np_random.uniform(low=-high, high=high)
        if self.close_to_goal == True:
            self.state = self.np_random.uniform(low=self.goal_state-0.5, high=self.goal_state+0.5)
        else:
            self.state = self.np_random.uniform(low=self.low_env_range, high=self.high_env_range)
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None
        self._render_reset()
        return self._get_obs()


    def _render_reset(self):
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")

    def _get_obs(self):
        theta, thetadot = self.state
        #self.collected_states.append(self.state)
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        # return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        x = self.state
        u = self.action_u
        x_noaction_local = self.x_noaction

        print ("no action:", x_noaction_local)
        print ("with action:", x)
        print ('action:', u)
        for i, m, k in [(x, 'o', 'green'), (x_noaction_local, '^', 'blue')]:
            self.ax.scatter3D(i[0], i[1],s=10,c=k,marker=m, alpha=0.5)
        #self.ax.scatter3D(x_noaction_local[0], x_noaction_local[1], x_noaction_local[2], s=10, c='blue', alpha=0.5)
        plt.title('pendulum attractor')
        plt.draw()
        #plt.show(block=False)
        #self.collected_states = list()
        plt.savefig('pendulum_ppo_2d.png')

#empowerment guy