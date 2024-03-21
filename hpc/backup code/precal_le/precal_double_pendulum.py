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

# g=9.8, l=1.0, m₁=10.0, m₂=1.0
# stateSpaceBox =[theta1,velocity1,theta2,velocity2]
# stateSpaceBox = [0 2pi;      -8 +8; 0 +2pi;        -8 8]
dyn = {"g":9.8, "m2": 1.0, "l": 1.0, "m1": 1.0}

v1 = np.array([1, 0, 0, 0], dtype=np.float32)
v2 = np.array([0, 1, 0, 0], dtype=np.float32)
v3 = np.array([0, 0, 1, 0], dtype=np.float32)
v4 = np.array([0, 0, 0, 1], dtype=np.float32)
x_dot = []
x_norm = []

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


    # g = np.array([0,
    #               (-1.25 - 0.5*np.cos(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))),
    #               0,
    #               1 / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
    #              ], dtype=np.float32)
    # return f + g*0
    return f

def linearized_double_pendulum (identity, dyn, x0):
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
    x = np.empty(shape=(len(x0),T), dtype=np.float32)
    v1_prime = np.empty(shape=(len(x0),T), dtype=np.float32)
    v2_prime = np.empty(shape=(len(x0),T), dtype=np.float32)
    v3_prime = np.empty(shape=(len(x0),T), dtype=np.float32)
    v4_prime = np.empty(shape=(len(x0),T), dtype=np.float32)
    x[:, 0] = x0
    v1_prime[:, 0] = v1
    v2_prime[:, 0] = v2
    v3_prime[:, 0] = v3
    v4_prime[:, 0] = v4
    le = np.array([0,0,0,0], dtype=np.float32)
    for i in range(1,T):

        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1])

        v1_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v1_prime[:, i-1], x[:, i-1])
        v2_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v2_prime[:, i-1], x[:, i-1])
        v3_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v3_prime[:, i-1], x[:, i-1])
        v4_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v4_prime[:, i-1], x[:, i-1])
#         print (v4_prime[:, i])
#         input()

        norm1 = np.linalg.norm(v1_prime[:, i])
        v1_prime[:, i] = v1_prime[:, i]/norm1

        GSC1 = np.dot(v2_prime[:, i], v1_prime[:, i])

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

        le = le + np.log2(np.array([norm1,norm2,norm3,norm4]))

#         if ( i % 100 == 0 ):
#             print ('log2:', np.log2(np.array([norm1,norm2,norm3]))/(i*dt))
#             print ('cum:', cum/(i*dt))


    return x, le/(T*dt)


def cal_le (x):
    _, le = f_t(dyn, double_pendulum, linearized_double_pendulum, 0.01, x, 300)
    return le
# precal_dp_15: 51
# precal_dp_16: 21
l2x = np.linspace(-np.pi ,  np.pi, 11)
l2y = np.linspace(-4 ,  4, 11)
l1x = np.linspace(-np.pi/2, 3*np.pi/2, 11)
l1y = np.linspace(-4 ,  4, 11)
X = np.array(np.meshgrid(l1x,l1y,l2x,l2y))


X_reshaped = X.T.reshape(X.T.shape[0]*X.T.shape[1]*X.T.shape[2]*X.T.shape[3],4)
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

np.save('/home/015970994/masterchaos/precal_le/Double_pendulum/precal_dp_10_300_001_fixedGS', zs)
np.save('/home/015970994/masterchaos/precal_le/Double_pendulum/precal_dp_points_10_300_001_fixedGS', X_reshaped)