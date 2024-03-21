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
from scipy import spatial


plt.style.use('fivethirtyeight')

class pendulum_le(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        super(pendulum_le, self).__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")
        self.close_to_goal = False
        self.infinite = False
        self.precal_le_  = np.load("/home/015970994/masterchaos/precal_le/Pendulum/precal_pendulum_20_0001_700.npy")
        self.precal_points = np.load("/home/015970994/masterchaos/precal_le/Pendulum/precal_pendulum_points_20_0001_700.npy")
        self.get_spatial = spatial.KDTree(self.precal_points)
        self.goal_state = np.array([np.pi, 0.0])
        self.max_speed = 8
        self.max_torque = 4.0
        self.T = 500
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.1
        self.dyn = {"g":self.g , "m":self.m, "l": self.l}
        self.alpha = 1
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

    def linearized_simple_pendulum (self, x0, dyn, y_pendulum):
        g = dyn['g'] 
        l = dyn['l']
        #print (x0)
        return np.array([0*x0[0] + 1*x0[1],
                        (-g/l)*np.cos(y_pendulum[0])*x0[0] + 0*x0[1]])

    def RungeKutta (self, dyn, f, dt, x0, action):

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x, k4
    
    def RungeKutta_linearized (self, dyn, f, dt, x0, y):
        k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
        k2 = f(x0+0.5*k1*dt,dyn, y)
        k3 = f(x0 + 0.5*k2*dt, dyn, y)
        k4 = f(x0 + k3*dt, dyn, y)
        
        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x


    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x, v = self.RungeKutta(dyn, f, dt, x0, action)
        #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
        return x, v
    
    # def f_x (self, dyn, f, dt, x0, action):
    #     g = dyn['g'] 
    #     l = dyn['l']
    #     m = dyn['m']
    #     #print (x0)
    #     # print (action)
    #     return x0 + np.array([x0[1] , (-g/l)*np.sin(x0[0]) + self.alpha * (1.0 / m) * action[0]])*dt
    
    def precal_le(self, s):
        
        index = self.get_spatial.query(s)[1]
        le = self.precal_le_[index]
        le = le[0]
        return le

    def le_reward (self, s , action):
        T = self.T
        v1 = np.array([1, 0], dtype=np.float32)
        v2 = np.array([0, 1], dtype=np.float32)
        cum = np.array([0,0], dtype=np.float32)

        x = np.empty(shape=(len(s),T), dtype=np.float32)
        v1_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v2_prime = np.empty(shape=(len(s),T), dtype=np.float32)

        x[:, 0] = s
        v1_prime[:, 0] = v1
        v2_prime[:, 0] = v2

        for i in range(1,T):
            x[:, i], _ = self.RungeKutta(self.dyn, self.simple_pendulum, 0.001, x[:, i-1], np.array([0]))
            
            v1_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_simple_pendulum, 0.001, v1_prime[:, i-1], x[:, i-1])
            v2_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_simple_pendulum, 0.001, v2_prime[:, i-1], x[:, i-1])
            
            norm1 = np.linalg.norm(v1_prime[:, i])
            v1_prime[:, i] = v1_prime[:, i]/norm1
            
            GSC1 = np.dot(v1_prime[:, i], v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]
            
            norm2 = np.linalg.norm(v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i]/norm2
            

            cum = cum + np.log2(np.array([norm1,norm2]))
    #         print("total time taken this loop: ", end_time - start_time)
        cum = cum/(T*self.dt)
        return max(cum)


    
    def step(self, u):
        x = self.state  # th := theta

        dyn = self.dyn
        f = self.simple_pendulum
        dt = self.dt
        # u = np.clip(u, -self.max_torque, self.max_torque)
        newx, v = self.f_x ( dyn, f, dt, x, u)
        # newx[0] = wrap(newx[0], 0, 2*np.pi)
        # newx[1] = bound(newx[1], -8, 8)
        # newx = np.clip(newx, self.low_env_range, self.high_env_range)
        # clip velocity
        # + 0.001 * (u**2)
        # 1. maybe NN cant work with angle, use cos and sin instead
        # 2. maybe we need to slight panelize action
        # newx[1] = np.clip(newx[1], self.low_env_range[1], self.high_env_range[1])
        # self.cost = - np.linalg.norm(x - self.goal_state) ** 2 + 0.1 * (np.linalg.norm(v)**2) + 0.001 * (np.linalg.norm(u)**2)
        self.cost = self.precal_le(x)

        terminated = False
        # if self.infinite == False:
        #     if x <=
        #     terminated = np.linalg.norm(x - self.goal_state) <= self.sphere_R
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
def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)
#empowerment guy