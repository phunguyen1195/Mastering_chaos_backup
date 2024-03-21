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

class cartpole_le(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        super(cartpole_le, self).__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")
        self.close_to_goal = False
        self.infinite = False
        self.T = 350
        self.max_torque = 2.0
        self.dt = 0.1
        self.dyn = {"g":9.8, "m2": 1.0, "l": 1.0, "m1": 10.0}
        self.alpha = 10
        self.render_mode = render_mode
        self.precal_le_ = np.loadtxt("precal_le_02.txt", delimiter=',')
        self.precal_points = np.load("points_02.npy")
        self.get_spatial = spatial.KDTree(self.precal_points)
        self.trajectory_collection = []
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.high_env_range = np.array([2, 1, 2*np.pi, 8], dtype=np.float32)
        self.low_env_range = np.array([-2,-1, 0, -8], dtype=np.float32)
        high = np.array([2.0, 1.0, 1.0, 1.0, 8.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape = (1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape = (5,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cartpole (self, x0, dyn, action):
        g = dyn['g'] 
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        #print (x0)
        f = np.array([x0[1],
                        (0.011*(x0[3]**2)*np.sin(x0[2]) + 0.098*np.cos(x0[2])*np.sin(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22),
                        x0[3],
                        (-1.96*np.sin(x0[2]) - 0.01*(x0[3]**2)*np.cos(x0[2])*np.sin(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22)
                        ], dtype=np.float32)
        g = np.array([0,
                    0.11 / (0.01*(np.cos(x0[2])**2) - 0.22),
                    0,
                    (-0.1*np.cos(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22)
            
        ], dtype=np.float32)
        return f + g*self.alpha*action[0]
    
    def linearized_cartpole (self, x0, dyn, y_cartpole):
        g = dyn['g'] 
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        led_cartpole = np.array([[0, 1, 0, 0],
                                [0, 0, (0.098*(np.cos(y_cartpole[2])**2) 
                                        + 0.011*(y_cartpole[3]**2)*np.cos(y_cartpole[2]) 
                                        - 0.098*(np.sin(y_cartpole[2])**2)) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22) 
                                + 0.02*((0.011*(y_cartpole[3]**2)*np.sin(y_cartpole[2]) 
                                        + 0.098*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])) / 
                                        ((0.01*(np.cos(y_cartpole[2])**2) - 0.22)**2))*np.cos(y_cartpole[2])*np.sin(y_cartpole[2]), 
                                (0.022*np.sin(y_cartpole[2])*y_cartpole[3]) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22)],
                                [0, 0, 0, 1],
                                [0, 0, (0.01*(np.sin(y_cartpole[2])**2)*(y_cartpole[3]**2) - 1.96*np.cos(y_cartpole[2]) - 0.01*(np.cos(y_cartpole[2])**2)*(y_cartpole[3]**2)) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22) + 0.02*((-1.96*np.sin(y_cartpole[2]) - 0.01*(y_cartpole[3]**2)*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])) / ((0.01*(np.cos(y_cartpole[2])**2) - 0.22)**2))*np.cos(y_cartpole[2])*np.sin(y_cartpole[2]), (-0.02*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])*y_cartpole[3]) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22)]
                                ], dtype=np.float32)
        return np.dot(x0,led_cartpole)

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
    

    def le_reward (self, s , action):
        T = self.T
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)
        v3 = np.array([0, 0, 1, 0], dtype=np.float32)
        v4 = np.array([0, 0, 0, 1], dtype=np.float32)
        cum = np.array([0,0,0,0], dtype=np.float32)

        x = np.empty(shape=(len(s),T), dtype=np.float32)
        v1_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v2_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v3_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v4_prime = np.empty(shape=(len(s),T), dtype=np.float32)

        x[:, 0] = s
        v1_prime[:, 0] = v1
        v2_prime[:, 0] = v2
        v3_prime[:, 0] = v3
        v4_prime[:, 0] = v4

        for i in range(1,T):

            x[:, i], _ = self.RungeKutta(self.dyn, self.cartpole, 0.001, x[:, i-1], np.array([0,0]))
            v1_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_cartpole, 0.001, v1_prime[:, i-1], x[:, i-1])
            v2_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_cartpole, 0.001, v2_prime[:, i-1], x[:, i-1])
            v3_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_cartpole, 0.001, v3_prime[:, i-1], x[:, i-1])
            v4_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_cartpole, 0.001, v4_prime[:, i-1], x[:, i-1])
            
            norm1 = np.linalg.norm(v1_prime[:, i])
            v1_prime[:, i] = v1_prime[:, i]/norm1
            
            GSC1 = np.dot(v1_prime[:, i], v2_prime[:, i])
            
            v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]
            norm2 = np.linalg.norm(v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i]/norm2
            
            GSC2 = np.dot(v3_prime[:, i], v1_prime[:, i])
            GSC3 = np.dot(v3_prime[:, i], v2_prime[:, i])
            
            v3_prime[:, i] = v3_prime[:, i] - GSC2*v1_prime[:, i] - GSC3*v2_prime[:, i]
            norm3 = np.linalg.norm(v3_prime[:, i])
            v3_prime[:, i] = v3_prime[:, i]/norm3
            
            GSC4 = np.dot(v4_prime[:, i], v1_prime[:, i])
            GSC5 = np.dot(v4_prime[:, i], v2_prime[:, i])
            GSC6 = np.dot(v4_prime[:, i], v3_prime[:, i])
            
            v4_prime[:, i] = v4_prime[:, i] - GSC4*v1_prime[:, i] - GSC5*v2_prime[:, i] - GSC6*v3_prime[:, i]
            norm4 = np.linalg.norm(v4_prime[:, i])
            
            v4_prime[:, i] = v4_prime[:, i]/norm4
            
            cum = cum + np.log2(np.array([norm1,norm2,norm3,norm4]))

        cum = cum/(T*self.dt)

        return max(cum) 
    
    def precal_le(self, s):
        
        index = self.get_spatial.query(s)[1]
        le = self.precal_le_[index]
        return le
    
    def sparse_reward(self, s):
        if np.linalg.norm(np.array([np.sin(s[2]),s[3]]))**2 - np.linalg.norm(np.array([np.sin(np.pi),0]))**2 <= 0.1:
            r = 1
        else:
            r = 0
        return r

    def step(self, u):
        x = self.state  # th := theta

        dyn = self.dyn
        f = self.cartpole
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)
        newx, v = self.f_x ( dyn, f, dt, x, u)

        newx[1] = np.clip(newx[1], self.low_env_range[1], self.high_env_range[1])
        newx[3] = np.clip(newx[3], self.low_env_range[3], self.high_env_range[3])
        # self.cost = self.le_reward(x, u) + 0.001 * (u[0]**2)
        # self.cost = self.precal_le(x) + 0.001 * (u[0]**2)
        self.cost = self.precal_le(x)
        terminated = False

        self.state = newx

        return self._get_obs(), self.cost, terminated, {'velocity': x, 'action': u, 'data': np.array([x[2],x[3]])}

    def reset(self):
        #put even closer to reward
        # high = np.array([np.pi, 1.0])
        # self.state = self.np_random.uniform(low=-high, high=high)
        if self.close_to_goal == True:
            sample_state = np.array([np.pi, 0.0])
            self.state = self.np_random.uniform(low=sample_state-0.5, high=sample_state+0.5)
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
        position, velocity, theta, thetadot = self.state
        #self.collected_states.append(self.state)
        return np.array([position, velocity, np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
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