import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)
w = 10.0
b = 1.0


def logistic_function(x: np.ndarray, w: float, b: float):

    m = x.shape[0]
    y = np.empty(x.shape)

    for i in range(m):
        f_wb = w * x[i] + b
        e_term = pow(np.e, -f_wb)
        y[i] = 1 / (1 + e_term)

    return y


y = logistic_function(x, w, b)

plt.plot(x, y, c="hotpink", label="logistic")
plt.title("Logistic Function")
plt.ylabel("y")
plt.xlabel("x")
plt.show()
