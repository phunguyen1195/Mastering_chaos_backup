import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from os import path
from typing import Optional

class LorenzEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, dt=0.01):
        self.collected_states = list()
        self.goal_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # self.goal_state = np.array([-8.485, -8.485, 27], dtype=np.float32)
        # self.goal_state = np.array([8.485, 8.485, 27], dtype=np.float32)

        self.infinite = True
        self.T = 500
        self.close_to_goal = True
        self.sphere_R = 3
        self.sigma = 10
        self.r = 28
        self.b = 8/3
        self.C = ((self.b ** 2) * ((self.sigma + self.r) ** 2)) / (4 * (self.b - 1))
        self.alpha = [100, 100, 100]
        self.dt = 0.01
        self.dyn = {"sigma":self.sigma , "R":self.r, "b": self.b}
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.action_range_high = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        self.action_range_low = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
        self.high_env_range = np.array([39.0, 39.0, 77.0], dtype=np.float32)
        self.low_env_range = np.array([-39.0, -39.0, -0.01], dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_range_low, high=self.action_range_high, shape = (3,),dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_env_range, high=self.high_env_range, shape = (3,),dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def lorenz (self, x0, dyn, action):
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        # print ('action in lorenz:', self.alpha*action)
        # print ('x0:', x0)
        return np.array([sigma * (y - x) + self.alpha[0]*action[0], 
                         x * (R - z) - y + self.alpha[1]*action[1], 
                         x * y - b * z + self.alpha[2]*action[2]])

    def RungeKutta (self, dyn, f, dt, x0, action):

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x
    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x = self.RungeKutta(dyn, f, dt, x0, action)
        #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
        return x
        
    def step(self, u):
        x = self.state
        dyn = self.dyn
        f = self.lorenz
        dt = self.dt
        newx = self.f_x ( dyn, f, dt, x, u)
        self.state = newx
        return self._get_obs(), 1, False, False, {}

    def reset(self, *, seed: Optional[int] = None, goal: Optional[list] = None):
        super().reset(seed=seed)
        if goal is not None:
            self.goal_state = np.array(goal)
            high = np.array([self.goal_state[0] + 1,self.goal_state[1]+ 1,self.goal_state[2]+ 1], dtype=np.float32)
            low = np.array([self.goal_state[0]-1,self.goal_state[1]-1, self.goal_state[2]- 1], dtype=np.float32)  # We enforce symmetric limits.
            # print ('high:', high )
            # print ('low:', low )
            self.state = self.np_random.uniform(low=low, high=high)
            # print (self.state)
        else:
            high = self.high_env_range
            low = self.low_env_range  # We enforce symmetric limits.
            # print ('high:', high )
            # print ('low:', low )
            y3 = self.np_random.uniform(low=low[2], high=high[2])
            new_C = self.C - np.square(y3 - self.r - self.sigma)
            y2 = self.np_random.uniform(low=-np.sqrt(new_C), high=np.sqrt(new_C))
            y23 = np.square(y2) + np.square(y3 - self.r - self.sigma)
            new_C  = self.C - y23
            y1 = self.np_random.uniform(low=-np.sqrt(new_C), high=np.sqrt(new_C))
            # print (np.square(y1) + np.square(y2) + np.square(y3 - self.r - self.sigma))
            self.state = np.array([y1,y2,y3])
        # print ('state:', self.state )
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        obsv = self.state
        #self.collected_states.append(self.state)
        
        return np.array(obsv, dtype=np.float32)