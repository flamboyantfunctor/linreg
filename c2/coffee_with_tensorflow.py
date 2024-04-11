import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from coffee_data import load_coffee_data


X, Y = load_coffee_data()

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)


goods = [Xn[i] for i in range(len(X)) if Y[i] == 1]
bads = [Xn[i] for i in range(len(X)) if Y[i] == 0]


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=3, activation="sigmoid"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
)

model.fit(Xn, Y, epochs=5)

plt.scatter(*zip(*goods), marker="x", c="r", label="Good Roast")
plt.scatter(*zip(*bads), marker="o", c="b", label="Bad Roast")
plt.legend()
plt.show()
