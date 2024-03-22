import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from exercise1 import RK_4th_order

# initial conditions
masses = np.array([3., 4., 5.])
position = np.array([[1., 3.], [-2., -1.], [1., -1.]])
velocity   = np.array([[0.,0.],[0.,0.],[0.,0.]])
#N-body units
G = 1.0   
t0 = 0.0
tf = 70.0
h0 = 0.1

# adaptive step size
def h_adaptive(r, h0):
    r12 = np.linalg.norm(r[0] - r[1])**-2.
    r23 = np.linalg.norm(r[1] - r[2])**-2.
    r31 = np.linalg.norm(r[0] - r[2])**-2.
    denominator = r12 + r23 + r31
    return h0/denominator

# equations of the inertial motion of three bodies:
def acceleration_three_body():
    acc = np.zeros((3,2), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                r = np.linalg.norm(position[i] - position[j])
                temp = - G*masses[j]*(position[i] - position[j]) / r**3.
                acc[i,:] += temp
    return acc





