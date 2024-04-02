import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

x = np.array([[200.0, 17.0]])

layer_1 = Dense(units=3, activation="sigmoid")
a_1 = layer_1(x)

layer_2 = Dense(units=1, activation="sigmoid")
a_2 = layer_2(a_1)

if a_2 >= 0.5:
    print(1)
else:
    print(0)
