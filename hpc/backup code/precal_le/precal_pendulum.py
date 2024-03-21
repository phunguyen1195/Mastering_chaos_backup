import multiprocessing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fractions import Fraction
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fractions import Fraction
import time

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

dyn_pendulum = {"g":9.81, "m": 1.0, "l": 1.0}
x0 = np.array([0.1, 0.0])
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 1.0])
x_dot = []
x_norm = []
# cum = np.array([0,0,0])

# def cosin 

def pendulum (x0, dyn):
    g = dyn['g'] 
    l = dyn['l']
    #print (x0)
    return np.array([x0[1], (-g/l)*np.sin(x0[0])])


def linearized_pendulum (x0, dyn, y_pendulum):
    g = dyn['g'] 
    l = dyn['l']
    #print (x0)
    x = y_pendulum[0]
    y = y_pendulum[1]
    pre_dot = np.array([[0, 1],
                    [(-g/l)*np.cos(x), 0],
                    ])
    af_dot = np.dot(pre_dot, x0)
    return af_dot



def RungeKutta (dyn, f, dt, x0):
    k1 = f(x0, dyn) #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1*dt,dyn)
    k3 = f(x0 + 0.5*k2*dt, dyn)
    k4 = f(x0 + k3*dt, dyn)
    
    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) *dt
    return x

def RungeKutta_linearized (dyn, f, dt, x0, y):
    k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
    k2 = f(x0+0.5*k1*dt,dyn, y)
    k3 = f(x0 + 0.5*k2*dt, dyn, y)
    k4 = f(x0 + k3*dt, dyn, y)
    
    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

    return x

def f_t (dyn, f, linearized_f, dt, x0, T):
    x = np.empty(shape=(len(x0),T))
    v1_prime = np.empty(shape=(len(x0),T))
    v2_prime = np.empty(shape=(len(x0),T))
    x[:, 0] = x0
    v1_prime[:, 0] = v1
    v2_prime[:, 0] = v2
    cum = np.array([0,0])
    
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1])
        
        v1_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v1_prime[:, i-1], x[:, i-1])
        v2_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v2_prime[:, i-1], x[:, i-1])
        
        norm1 = np.linalg.norm(v1_prime[:, i])
        v1_prime[:, i] = v1_prime[:, i]/norm1
        
        GSC1 = np.dot(v1_prime[:, i], v2_prime[:, i])
        v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]
        
        norm2 = np.linalg.norm(v2_prime[:, i])
        v2_prime[:, i] = v2_prime[:, i]/norm2
        

        cum = cum + np.log2(np.array([norm1,norm2]))
#         print("total time taken this loop: ", end_time - start_time)
    cum = cum/(T*dt)
    return x,cum


def cal_le (x):
    _, le = f_t(dyn_pendulum, pendulum, linearized_pendulum, 0.001, x, 700)
    return le

lx = np.linspace(-np.pi, 2*np.pi, 101)
ly = np.linspace(-8, 8, 101)
X = np.array(np.meshgrid(lx,ly))


X_reshaped = X.T.reshape(X.T.shape[0]*X.T.shape[1],2)
# X_reshaped = np.append(X_reshaped,np.array([[np.pi/2,0,0,0]]),axis=0)
# new_test_sin = np.array([np.sin(X_reshaped.T[0]),X_reshaped.T[1]])
le_list = []
points = list(X_reshaped)
# print ('I am here')
with ProcessPoolExecutor(max_workers=cores-1) as executor:
    for r in executor.map(cal_le, points, chunksize=10):
        # print (r)
        le_list.append(r)
# print (points[0])
# for i in points:
#   %time le_list.append(cal_le (i))

zs = np.array(le_list)

np.save('/home/015970994/masterchaos/precal_le/Pendulum/precal_pendulum_pi_0001_700_2', zs)
np.save('/home/015970994/masterchaos/precal_le/Pendulum/precal_pendulum_points_pi_0001_700_2', X_reshaped)