import numpy as np

def calc_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        i = it.multi_index
        tmp = x[i]

        x[i] = tmp + h
        y1 = f(x)

        x[i] = tmp - h
        y2 = f(x)

        x[i] = tmp
        grad[i] = (y1 - y2) / (2 * h)

        it.iternext()

    return grad