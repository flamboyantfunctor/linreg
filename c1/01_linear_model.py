# %% [markdown]
# # A Basic Linear Regression example

import numpy as np
import matplotlib.pyplot as plt


# Inputs
x = np.array([1.0, 2.0])
# Outputs
y = np.array([300.0, 500.0])

# Initial weight and bias
w = 200
b = 100


# Model
def compute_model(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Computes the prediction of a linear model"""
    f = np.dot(x, w) + b
    return f


f = compute_model(
    x,
    w,
    b,
)

# Make a prediction on a unknown input
xs = np.array([1.2, 1.3, 1.9])
ys = compute_model(xs, w, b)
# Plot the model
plt.plot(x, f, c="b", label="prediction model")
# Plot the training data points
plt.scatter(x, y, marker="x", c="hotpink", label="actual values")
# Plot the unknown data points
plt.scatter(xs, ys, marker="o", c="hotpink", label="predicted value")
# Plot Configuration
plt.title("Housing Prices")
plt.ylabel(f"Price in $10^3$ dollars")
plt.xlabel(f"Size in $10^3$ sqft")
plt.legend()
plt.show()

# %%
