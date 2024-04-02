import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.keras.activations import sigmoid


X_train = np.array([[1.0], [2.0]], dtype=np.float32)  # (size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)  # (price in 1000s of dollars)


linear_layer = Dense(
    units=1,
    activation="linear",
)

a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)

w, b = linear_layer.get_weights()
print(f"Weight: {w} , Bias: {b}")

set_w = np.array([[200]])
set_b = np.array([100])

linear_layer.set_weights([set_w, set_b])
w, b = linear_layer.get_weights()
print(f"Weight: {w} , Bias: {b}")


# fig, ax = plt.subplots(1, 1)
# ax.scatter(X_train, Y_train, marker="x", c="r", label="Data Points")
# ax.legend(fontsize="small")
# ax.set_ylabel("Price (in 1000s of dollars)", fontsize="small")
# ax.set_xlabel("Size (1000 sqft)", fontsize="small")
# plt.show()
