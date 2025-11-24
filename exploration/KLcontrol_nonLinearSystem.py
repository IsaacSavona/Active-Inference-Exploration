import numpy as np

# time
tau = 0.05
k = 5000

# physical system
m = 0.1 # kg
L = 1.0 # m
M = 1.0 # kg
g = 9.8 # m/sec^2
sigma = 5 # N

# cost function weights
q1 = 7   # m^-1
q2 = 2.5 # m/s
q3 = 7.0 # rad^-1
q4 = 2.5 # s/rad

X = np.empty([k, 4]) # 5000 samples and 4 states
x0 = np.array([2, 0, 0.5, 0])

def h1(theta, theta_dot, u):
    return  -m * L * theta_dot**2 * np.sin(theta) \
        + m * g * np.sin(theta) * np.cos(theta) + u \
    / (M + m * np.sin(theta)**2)

def h2(theta, theta_dot, u):
    return  (1 / L) * h1(theta, theta_dot, u) * np.cos(theta) \
        + g * sin(theta)

def f(x, u, k):
    x_next = x
    x_next[0] = x[0] + tau * x[1]
    x_next[1] = x[1] + tau * h1(theta, theta_dot, u)
    x_next[2] = theta + tau * x[3]
    x_next[3] = x[3] + tau * h2(theta, theta_dot, u)
    return x_next

def l(x):
    return q1 * x[0] + q2 * x[1] + q3 * x[2] + q4 * x[3]

def Z(k, x):
    b = np.exp(-np.sum(l(x)))
    return expectation_value(b)