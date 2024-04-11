import numpy as np


def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return 1 if result >= 0.5 else 0


X = np.array([[200.0, 17.0]])

W = np.array([[1, -3, 5], [-2, 4, -6]])

B = np.array([[-1, 1, 2]])


def dense(A_IN, W, B):
    Z = np.matmul(A_IN, W) + B
    A_OUT = np.vectorize(sigmoid)(Z)
    return A_OUT


result = dense(X, W, B)
print(result)
