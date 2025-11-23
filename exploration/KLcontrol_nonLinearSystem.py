import numpy as np

tau = 0.05
k = 5000
X = np.empty([k, 4]) # 5000 samples and 4 states

def f(x, u, k):
    x_next = x
    x_next[0] = x[0] + tau*x[1]
    ### ...
    return x_next