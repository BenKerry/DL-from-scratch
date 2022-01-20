import numpy as np

def mean_squared_error(y, t):
    if y.ndim == 1:
        t = y.reshape(1, t.size)
        y = y.reshape(1, y.size)

    return (1 / y.shape[0]) * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = y.reshape(1, t.size)
        y = y.reshape(1, y.size)

    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]