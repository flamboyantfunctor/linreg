import numpy as np
import matplotlib.pyplot as plt

# Inputs: Training Examples with 2 features
X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])

# Corresponding labels for classification
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# I just love list comprehensions.. Shoutout to Haskell!
ones = [X[i] for i in range(len(X)) if y[i] == 1]
zeros = [X[i] for i in range(len(X)) if y[i] == 0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

"""Decision Boundary"""
# We trained the model and got these values: w_0 = 1, w_0 = 1, b = -3


# This is *-moves are f-ing elegant 😳 i don't get it yet...
ax.scatter(*zip(*ones), marker="x", c="r", label="y=1")
ax.scatter(*zip(*zeros), marker="o", c="b", label="y=0")

ax.axis([0.0, 4.0, 0.0, 4.0])
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend()
plt.show()
