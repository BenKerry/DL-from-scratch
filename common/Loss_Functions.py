import numpy as np

def mean_squared_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    return np.sum((y - t) ** 2) / y.shape[0]

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

if __name__ == "__main__":
    print(cross_entropy_error(np.array([0.1, 0.5, 0.4]), np.array([0, 1, 0])))