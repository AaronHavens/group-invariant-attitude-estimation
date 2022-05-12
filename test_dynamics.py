from dynamics import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

omega = np.array([[-.1], [0.5], [-.5]])
def omega_fn(t):
    omega1 = np.sin(0.7*t)
    omega2 = 0.7*np.sin(0.5*t + np.pi)
    omega3 = 0.5*np.sin(0.3*t + np.pi/3)
    return np.array([[omega1],[omega2],[omega3]])

R = np.eye(3)
T = 100.0
dt = 1e-4
N = int(T//dt)
e = np.array([[0],[0],[1]])
P = []
t = 0
for i in tqdm(range(N)):
    p_t = pose(e, R)
    P.append(p_t)
    omega_t = omega_fn(t)
    #R = exp_step(R, omega_t, dt)
    R = euler_step(R, omega_t, dt)
    t += dt
    #print(np.linalg.det(R), R.T@R, R@R.T)

P = np.asarray(P)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect(aspect=(1,1,1))
base = np.zeros((N,3))
r_lim = 1.0
globe(ax)
ax.axes.set_xlim3d(-r_lim, r_lim)
ax.axes.set_ylim3d(-r_lim, r_lim)
ax.axes.set_zlim3d(-r_lim, r_lim)
ax.scatter3D(e[0,0],e[1,0],e[2,0],c='r')
ax.plot3D(P[:,0,0], P[:,1,0], P[:,2,0])
#for i in range(N):
    #ax.scatter(P[i,0], P[i,1], P[i,2])
    #ax.quiver(base[i,0], base[i,1], base[i,2], P[i,0], P[i,1], P[i, 2])
    #plt.pause(0.0001)

plt.show()

 
