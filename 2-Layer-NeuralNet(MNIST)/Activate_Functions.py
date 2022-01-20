import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    max_val = np.max(x)
    exp_x = np.exp(x - max_val)
    exp_x_sum = np.sum(exp_x - max_val)

    return exp_x / exp_x_sum