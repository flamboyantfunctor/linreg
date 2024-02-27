import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([1.0, 2.0, 4.0])
y_train = np.array([300.0, 500.0, 600.0])

m = x_train.shape[0]

w = 200
b = 100


def compute_model_output(x, w, b):
    """Computes the prediction of a linear model"""

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


tmp_f_wb = compute_model_output(
    x_train,
    w,
    b,
)

plt.plot(x_train, tmp_f_wb, c="b", label="prediction model")
plt.scatter(x_train, y_train, marker="x", c="r", label="actual values")
plt.title("Housing Prices")
plt.ylabel("Price in 1000s dollars")
plt.xlabel("size in 1000s sqft")
plt.legend()
plt.show()


x_1 = 1.2
cost_1200sqft = w * x_1 + b

print(f"Price Prediction for a 1200sqft house: {cost_1200sqft} * 1000$")
