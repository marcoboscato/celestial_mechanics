import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from exercise1 import RK_4th_order

# initial conditions
masses = np.array([3., 4., 5.])
pos = np.array([[1., 3.], [-2., -1.], [1., -1.]])
vel = np.array([[0.,0.],[0.,0.],[0.,0.]])
#N-body units
G = 1.0   
t0 = 0.0
tf = 70.0
h0 = 0.1

def h_adaptive(r, h0):
    r12 = np.linalg.norm(r[0] - r[1])**-2.
    r23 = np.linalg.norm(r[1] - r[2])**-2.
    r31 = np.linalg.norm(r[0] - r[2])**-2.
    denominator = r12 + r23 + r31
    return h0/denominator

# RK4th order adaptive update
def RK_4th_order_adaptive(f, x0, t0, tf, h0):
    t = np.arange(t0, tf, h0)
    n = len(t)
    x = np.zeros([n, len(x0)], dtype=float)
    x[0] = x0
    h = h_adaptive(x[0][:6].reshape(3, 2), h0)

    for i in range(n-1):
        k1 = h*f(t[i], x[i])
        k2 = h*f(t[i] + h/2, x[i] + k1/2)
        k3 = h*f(t[i] + h/2, x[i] + k2/2)
        k4 = h*f(t[i] + h, x[i] + k3)

        x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4)/6.0

        h = h_adaptive(x[i+1][:6].reshape(3, 2), h0)

    return t, x

# Energy function for 3 bodies
def Energy_3bodies(x):
    r12 = np.linalg.norm(x[:,0:2] - x[:,2:4])
    r23 = np.linalg.norm(x[:,2:4] - x[:,4:6])
    r31 = np.linalg.norm(x[:,0:2] - x[:,4:6])
    Ekin = 0.5*(masses[0]*np.sum(x[:,6:8]**2, axis=1) + masses[1]*np.sum(x[:,8:10]**2, axis=1) + masses[2]*np.sum(x[:,10:12]**2, axis=1))
    Epot = G*(masses[0]*masses[1]/r12 + masses[1]*masses[2]/r23 + masses[0]*masses[2]/r31)
    E = Ekin - Epot
    return np.abs((E - E[0])/E[0])

# Acceleration function for 3 bodies
def acceleration_three_body(x):
    acc = np.zeros((3,2), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                r = np.linalg.norm(x[i] - x[j])
                temp = - G*masses[j]*(x[i] - x[j]) / r**3.
                acc[i,:] += temp
    return acc

# Function to integrate
def f(t, x):
    r = x[0:6].reshape(3, 2)
    v = x[6:12].reshape(3, 2)
    a = acceleration_three_body(r)
    return np.array(np.concatenate((v[0], v[1], v[2], a[0], a[1], a[2]), axis=None))


# RK4th FUNCTION
x = np.concatenate((pos, vel), axis=None)

time, orbit = RK_4th_order_adaptive(f, x, t0, tf, h0)

energy = Energy_3bodies(orbit)


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(orbit[:,0], orbit[:,1], label='Body 1')
plt.plot(orbit[:,2], orbit[:,3], label='Body 2')
plt.plot(orbit[:,4], orbit[:,5], label='Body 3')
plt.xlabel('x [N-body Units]')
plt.ylabel('y [N-body Units]')
plt.title('Burrau problem - RK45')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(time, energy, label='energy variation')
plt.xlabel('time [N-body Units]')
plt.ylabel('$|(E - E_{0})/E_{0}|$')
plt.title('Energy variation')
plt.grid()
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()


# RK4th WITH SCIPY

solution = solve_ivp(fun=f, t_span=[t0, tf], y0=x, method='RK45', dense_output=True, rtol=1e-12, atol=1e-12)
energy2 = Energy_3bodies(solution.y.T)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(solution.y[0], solution.y[1], label='mass=3')
plt.plot(solution.y[2], solution.y[3], label='mass=4')
plt.plot(solution.y[4], solution.y[5], label='mass=5')
plt.xlabel('x [N-body Units]')
plt.ylabel('y [N-body Units]')
plt.title('Burrau problem - RK45 (scipy)')
plt.xlim(-4, 4)
plt.ylim(-10, 5)
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(solution.t, energy2, label='energy variation')
plt.xlabel('time [s]')
plt.ylabel('$|(E - E_{0})/E_{0}|$')
plt.title('Energy variation')
plt.grid()
plt.legend(loc='lower left')

plt.tight_layout()





