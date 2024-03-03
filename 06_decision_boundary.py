import numpy as np
import matplotlib.pyplot as plt

# Inputs: Training Examples with 2 features
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])

# Corresponding labels for classification

y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
# ...reshape(-1, 1) -> [[0],[0],[0],[1],[1],[1] ]

# I just love list comprehensions.. Shoutout to Haskell!
ones = [X[i] for i in range(len(X)) if y[i] == 1]
zeros = [X[i] for i in range(len(X)) if y[i] == 0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

"""
Decision Boundary

z = w_0 * x_0 + w_1 * x_1 + b

We trained the model and got these values: w_0 = 1, w_0 = 1, b = -3

z = x_0 + x_1 - 3
x_0 + x_1 = 3
x_0 = 3 - x_1

"""
x_1 = np.arange(5)
x_0 = 3 - x_1

# This *-moves are f-ing elegant 😳 i don't get it yet...
"""
*ones                   -> unpacks *ones, so [3, 0.5], [2, 2], [1, 2.5]
zip(*ones)              -> zip([3, 0.5], [2, 2], [1, 2.5]) => [3, 2, 1], [0.5, 2, 2.5]
(*zip(*ones))           -> *zip.. unpacks the two iters that zip(*ones) returns
"""

# Plotting the training examples with label y=1
ax.scatter(*zip(*ones), marker="x", c="r", label="y=1")
# Plotting the training examples with label y=1
ax.scatter(*zip(*zeros), marker="o", c="b", label="y=0")
# Plotting the decision boundary
ax.plot(x_0, x_1)
# Configuring the figure
ax.fill_between(x_0, x_1, alpha=0.1)
ax.axis([0.0, 4.0, 0.0, 4.0])
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
plt.subplots_adjust(left=0.10, bottom=0.10)
plt.legend()

# Show plot
plt.show()
