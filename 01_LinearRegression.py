import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

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


x_1 = 1.2
y_1_hat = w * x_1 + b
print(f"Price Prediction for a 1200sqft house: {y_1_hat} * 1000$")


plt.plot(x_train, tmp_f_wb, c="b", label="prediction model")
plt.scatter(x_train, y_train, marker="x", c="hotpink", label="actual values")
plt.scatter(x_1, y_1_hat, marker="o", c="hotpink", label="predicted value")
plt.title("Housing Prices")
plt.ylabel("Price in 1000s dollars")
plt.xlabel("size in 1000s sqft")
plt.legend()
plt.show()
