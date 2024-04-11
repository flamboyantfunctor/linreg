import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential

x = np.array([[200.0, 17.0]])

layer_1 = Dense(units=3, activation="sigmoid")
a_1 = layer_1(x)

layer_2 = Dense(units=1, activation="sigmoid")
a_2 = layer_2(a_1)


# Another (shorter) way...

x = np.array(
    [
        [200.0, 17.0],
        [120.0, 5.0],
        [425.0, 20.0],
        [212.0, 18.0],
    ]
)

y = np.array([1, 0, 0, 1])


model = Sequential(
    [Dense(units=3, activation="sigmoid"), Dense(units=1, activation="sigmoid")]
)
# model.compile(...)
# model.fit(x, y)
# model.predict(...)
