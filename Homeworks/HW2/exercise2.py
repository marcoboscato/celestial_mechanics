import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from exercise1 import RK_4th_order

# initial conditions
GM = 398600.4415 # km^3/s^2   (gravitational parameter of the Earth)
position = np.array([[0., 0.], [384400.0, 0.]])                         #position of the two bodies (x,y) in klometers
velocity = np.array([[0., 0.], [0.91647306922544, 0.91647306922544]])   #velocity of the two bodies (vx,vy) in km/s
particles = np.concatenate((position, velocity), axis=None)             #concatenate position and velocity of the two bodies as x(t) for RK_4th_order
                                                  
# time interval
t0 = 0
tf = 3e6
h = 1e2

# equations of the inertial motion of two bodies:
def acceleration_two_body(t, x):
    r12 = x[2:4] - x[0:2]
    denom = np.linalg.norm(r12)**3
    acc = GM*(x[2:4] - x[0:2]) / denom
    return np.array(np.concatenate((x[4:6], x[6:8], acc, -acc), axis=0))     # return the acceleration of the two bodies in 1D for RK_4th_order

# R-K of 4th order
time1, orbit1 = RK_4th_order(acceleration_two_body, particles, t0, tf, h)

# calculate the position of the Center of Mass
cm1 = np.empty([len(orbit1), 2], dtype=float)
for i in range(len(orbit1)):
    cm1[i] = np.sum(orbit1[i].reshape(-1, 2), axis=0)/2.0

def Energy(x):
    r12 = x[:,2:4] - x[:,0:2]
    E = 0.5*np.sum(x[:,4:6]**2, axis=1) + 0.5*np.sum(x[:, 6:8]**2, axis=1) - GM/np.linalg.norm(r12, axis=1)
    return np.abs(E - E[0])/E[0]

# calculate the energy of the system
energy1 = Energy(orbit1)

# plots
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(orbit1[:,0], orbit1[:,1], label='Body 1')
plt.plot(orbit1[:,2], orbit1[:,3], label='Body 2')
plt.plot(cm1[:,0], cm1[:,1], label='Center of Mass', linestyle='--', color='lightgrey', zorder=0)
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.title('Orbit of two bodies - RK4th order')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(time1, energy1, label='energy variation')
plt.xlabel('time [s]')
plt.ylabel('$E/E_{0}$')
plt.title('Energy variation')
plt.grid()
plt.legend(loc='lower left')

plt.tight_layout()




# ORBIT WITH SCIPY
sol = solve_ivp(fun=acceleration_two_body, t_span=[t0, tf], y0=particles, method='RK45', 
                dense_output=True, rtol=1e-6)
time2 = sol.t
print(sol.y.shape)

# calculate the position of the Center of Mass
cm2 = np.empty([sol.y.shape[1], 2], dtype=float)
for i in range(sol.y.shape[1]):
    x_cm2 = 0.5*(sol.y[0,i] + sol.y[2,i])
    y_cm2 = 0.5*(sol.y[1,i] + sol.y[3,i])
    cm2[i] = [x_cm2, y_cm2]

# calculate the energy of the system
energy2 = Energy(sol.y.T)

# plot the orbit
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(sol.y[0], sol.y[1], label='Body 1')
plt.plot(sol.y[2], sol.y[3], label='Body 2')
plt.plot(cm2[:,0], cm2[:,1], label='Center of Mass', linestyle='--', color='lightgrey', zorder=0)
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.title('Orbit of two bodies - RK45 (scipy)')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(time2, energy2, label='energy variation')
plt.xlabel('time [s]')
plt.ylabel('$E/E_{0}$')
plt.title('Energy variation')
plt.grid()
plt.legend(loc='lower left')

plt.tight_layout()
