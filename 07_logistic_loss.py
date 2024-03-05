import numpy as np
import matplotlib.pyplot as plt

# Inputs: Training Examples with 2 features
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])

# Ouputs: Corresponding input labels
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
# ...reshape(-1, 1) -> [[0],[0],[0],[1],[1],[1]]

# I just love list comprehensions.. Shoutout to Haskell!
ones = [X[i] for i in range(len(X)) if y[i] == 1]
zeros = [X[i] for i in range(len(X)) if y[i] == 0]


"""
Decision Boundary

z = w_0 * x_0 + w_1 * x_1 + b

We trained the model and got these values: w_0 = 1, w_0 = 1, b = -3

z = x_0 + x_1 - 3
x_0 + x_1 = 3
x_0 = 3 - x_1

"""
# Weights and bias
w_m1 = np.array([1, 1])
b_m1 = -3

w_m2 = np.array([1, 1])
b_m2 = -4

# Formula for the 1,1,-3 model
x_1_m1 = np.arange(10)
x_0_m1 = -x_1_m1 + 3
# Formula for the 1,1,-4 model
x_1_m2 = np.arange(10)
x_0_m2 = -x_1_m2 + 4


"""
Cost function:
"""


def compute_logistic_cost(X, y, w, b):
    """
    Computes tht total cost for the logistic regression

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m, _ = X.shape

    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = 1 / (1 + np.exp(-z_i))
        loss = -(y[i] * np.log(f_wb_i)) - (1 - y[i]) * np.log(1 - f_wb_i)
        cost += loss
    cost = cost / m
    return cost


text_m1 = f"Cost of model 1: {compute_logistic_cost(X, y, w_m1, b_m1)}"
text_m2 = f"Cost of model 2: {compute_logistic_cost(X, y, w_m2, b_m2)}"

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Elegant zipping and unpacking.. i don't get it yet
"""
*ones                   -> unpacks *ones, so [3, 0.5], [2, 2], [1, 2.5]
zip(*ones)              -> zip([3, 0.5], [2, 2], [1, 2.5]) => [3, 2, 1], [0.5, 2, 2.5]
(*zip(*ones))           -> *zip.. unpacks the two iters that zip(*ones) returns
"""
# Plotting the training examples with label y=1
ax.scatter(*zip(*ones), c="r", marker="x", label="y=1")
# Plotting the training examples with label y=1
ax.scatter(*zip(*zeros), c="b", marker="o", label="y=0")
# Plotting the decision boundary
ax.plot(x_0_m1, x_1_m1, c="y", linestyle="dashed", label="model 1")
ax.plot(x_0_m2, x_1_m2, c="g", linestyle="dashed", label="model 2")
# Configuring the figure
ax.axis([0.0, 4.0, 0.0, 4.0])
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.text(3, 2.75, text_m1, color="y", ha="center")
plt.text(3, 2.5, text_m2, color="g", ha="center")
plt.legend()
plt.show()
