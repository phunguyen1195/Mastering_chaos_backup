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

dyn_rossler = {"a": 0.15, "b": 0.2, "c": 10 }
x0 = np.array([10.0, 1.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])

fp1 = np.array([(dyn_rossler['c'] + np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/2,
               (- dyn_rossler['c'] - np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/(2*dyn_rossler['a']),
               (dyn_rossler['c'] + np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/(2*dyn_rossler['a'])])
fp2 = np.array([(dyn_rossler['c'] - np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/2,
               (- dyn_rossler['c'] + np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/(2*dyn_rossler['a']),
               (dyn_rossler['c'] - np.sqrt(dyn_rossler['c']**2 - 4*dyn_rossler['a']*dyn_rossler['b']))/(2*dyn_rossler['a'])])

# fp2 = np.array([])

x_dot = []
x_norm = []
le_array = []
# cum = np.array([0,0,0])

def Rossler (x0, dyn):
    a = dyn['a']
    b = dyn['b']
    c = dyn['c']
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([-y - z,
                     x + a*y,
                     b + z*(x-c)])


def linearized_rossler (x0, dyn, y_rossler):
    a = dyn['a']
    b = dyn['b']
    c = dyn['c']
    x = y_rossler[0]
    y = y_rossler[1]
    z = y_rossler[2]
    pre_dot = np.array([[0, -1, -1],
                        [1, a, 0],
                        [z,0,x-c]
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
    v3_prime = np.empty(shape=(len(x0),T))
    x[:, 0] = x0
    v1_prime[:, 0] = v1
    v2_prime[:, 0] = v2
    v3_prime[:, 0] = v3
    le = np.array([0,0,0])
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1])

        v1_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v1_prime[:, i-1], x[:, i-1])
        v2_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v2_prime[:, i-1], x[:, i-1])
        v3_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v3_prime[:, i-1], x[:, i-1])


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

        # print (le)
        le = le + np.log2(np.array([norm1,norm2,norm3]))
        le_array.append(le/(i*dt))
#         if ( i % 100 == 0 ):
#             print ('log2:', np.log2(np.array([norm1,norm2,norm3]))/(i*dt))
#             print ('cum:', cum/(i*dt))


    return x, le/(T*dt)


def cal_le (x):
    _, le = f_t(dyn_rossler, Rossler, linearized_rossler, 0.1, x, 1)
    return le

# lx = np.linspace(fp1[0] -2, fp1[0] +2, 21)
# ly = np.linspace(fp1[1] -2, fp1[1] +2, 21)
# lz = np.linspace(fp1[2] -2, fp1[2] +2, 21)
# X = np.array(np.meshgrid(lx,ly,lz))

lx = np.linspace(-10, 15, 31)
ly = np.linspace(-30, 10, 31)
lz = np.linspace(0, 30, 31)
X = np.array(np.meshgrid(lx,ly,lz))

X_reshaped = X.T.reshape(X.T.shape[0]*X.T.shape[1]*X.T.shape[2],3)
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

np.save('/home/015970994/masterchaos/precal_le/Rossler/precal_rossler_1_001_1000', zs)
np.save('/home/015970994/masterchaos/precal_le/Rossler/precal_rossler_points_1_001_1000', X_reshaped)