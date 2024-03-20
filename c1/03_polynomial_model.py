import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

x = np.arange(-1, 1, 0.1)

# Ominous Target Function - In reality we'll never know this function
y = 5 - 2 * (x**3)

# Create input data that has 3 polynomial features:
# np.c_ is a function that concatenates the specified values
X = np.c_[x**1, x**2, x**3]

# Assign some random weights.. We need 3 weights because we have 3 features
w = np.array([2, -0.8, 4])

# Assign a random bias value
b = 0


def compute_cost(X, y, w, b):
    """Computes the Cost (MSE-mean squared error) for given inputs, targets, weights and bias values"""
    m = len(y)
    h = X @ w + b
    error = h - y
    J = 1 / (2 * m) * np.sum(error**2)
    return J


def update_wb(X, y, w, b, alpha):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        dj_dw += ((X[i] @ w + b) - y[i]) * X[i]
        dj_db += (X[i] @ w + b) - y[i]

    dj_dw /= m
    dj_db /= m

    w -= alpha * dj_dw
    b -= alpha * dj_db
    return w, b


def fit(X, y, w, b, iterations, alpha):
    i = 0
    while i <= iterations:
        if i % 1000 == 0:
            print(f"Iterations:\t{i}\t\tCost:\t{compute_cost(X, y, w, b)}")
        w, b = update_wb(X, y, w, b, alpha)
        i += 1
    print(f"Optimal weights:\t{w}\t\tOptimal bias:\t{b} ")
    return w, b


model_w, model_b = fit(X, y, w, b, 20000, 1e-2)
plt.scatter(x, y, marker="x", c="hotpink", label="datapoints")
plt.plot(x, X @ model_w + model_b, label="model")
plt.legend()
plt.show()
