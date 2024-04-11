import numpy as np


def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return 1 if result >= 0.5 else 0


a_in = np.array([200.0, 17.0])

W = np.array([[1, -3, 5], [-2, 4, -6]])

b = np.array([-1, 1, 2])


def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out


a_out = dense(a_in, W, b)
print(a_out)
