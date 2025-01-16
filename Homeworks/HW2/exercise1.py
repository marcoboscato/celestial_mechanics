import numpy as np

# R-K of 4th order
def RK_4th_order(f, x0, t0, tf, h):
    t = np.arange(t0, tf, h)
    n = len(t)
    x = np.zeros([n, len(x0)], dtype=float)
    x[0] = x0

    for i in range(n-1):
        k1 = h*f(t[i], x[i])
        k2 = h*f(t[i] + h/2, x[i] + k1/2)
        k3 = h*f(t[i] + h/2, x[i] + k2/2)
        k4 = h*f(t[i] + h, x[i] + k3)

        x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4)/6.0

    return t, x