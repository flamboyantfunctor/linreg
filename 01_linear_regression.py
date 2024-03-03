import numpy as np
import matplotlib.pyplot as plt

# Training Data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# Initial weight and bias parameters
w = 200
b = 100


def compute_model(x, w, b):
    """Computes the prediction of a linear model"""

    m = x.shape[0]

    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


f_wb = compute_model(
    x_train,
    w,
    b,
)

# Make a prediction on a unknown input
x_1 = np.array([1.2, 1.3, 1.9])
p_1 = compute_model(x_1, w, b)
# Plot the linear model
plt.plot(x_train, f_wb, c="b", label="prediction model")
# Plot the training data points
plt.scatter(x_train, y_train, marker="x", c="hotpink", label="actual values")
# Plot the unknown data points
plt.scatter(x_1, p_1, marker="o", c="hotpink", label="predicted value")
# Plot Configuration
plt.title("Housing Prices")
plt.ylabel(f"Price in $10^3$ dollars")
plt.xlabel(f"Size in $10^3$ sqft")
plt.legend()
plt.show()
