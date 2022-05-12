import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def globe(ax, radius=1.0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    #for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)
    elev = 10.0
    rot = 80.0 / 180 * np.pi
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='grey', linewidth=0, alpha=0.2)
    #calculate vectors for "vertical" circle
    a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    b = np.array([0, 1, 0])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = '--')
    horiz_front = np.linspace(0, np.pi, 100)
    ax.plot(np.sin(horiz_front),np.cos(horiz_front),0,color='k', linestyle='--')
    #vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    #ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),color='k', linestyle = '--')
    #ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),color='k',linestyle='--')

def skew(a):
    a1 = a[0,0]
    a2 = a[1,0]
    a3 = a[2,0]
    A = np.array([  [0,-a3, a2],
                    [a3,0,-a1],
                    [-a2, a1, 0]])
    return A

def vex(A):
    a = np.array([[A[2,1]], [A[0,2]], [A[1,0]]])
    return a

def antisym_proj(B):
    A = 1/2*(B -  B.T)
    return A

def so3_norm_dist(R):
    return 1/4*np.trace(np.eye(3) - R)

def angle_axis_param(alpha, u):
    return np.eye(3) + np.sin(alpha)*skew(u) \
        + (1-np.cos(alpha))*np.linalg.matrix_power(skew(u), 2)

def rodriguez_param(rho):
    rho_norm = np.linalg.norm(rho)**2
    return 1/(1+rho_norm)*((1-rho_norm)*np.eye(3)+ 2*rho@rho.T + 2*skew(rho))

def rodriguez_proj(rho):
    rho_norm = np.linalg.norm(rho)**2
    return 2/(1+rho_norm)*skew(rho)

def so3_rodriguez_dist(rho):
    rho_norm = np.linalg.norm(rho)**2
    return rho_norm/(1+rho_norm)

def error_from_body(R, R_hat):
    return R.T@R_hat

def euler_step(R, omega, dt):
    return R + R@skew(omega)*dt

def exp_step(R, omega, dt):
    return R@expm(skew(omega)*dt)

def ode_ivp(R0, omega, T):
    A = skew(omega)
    R0 = R0.reshape(-1)
    def odefun(t, x):
        x = x.reshape([3,3])
        dx = x@A
        return dx.reshape(-1)

    sol = solve_ivp(odefun, [0, T], R0)

    return sol

def pose(e, R):
    return R@e


