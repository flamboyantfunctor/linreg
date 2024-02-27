import numpy as np
import time

a = np.zeros(4)
print(f"np.zeros(4) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros((4,)) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}")
a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}"
)

a = np.arange(4.0)
print(f"np.arange(4.0) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}")

a = np.random.rand(4)
print(f"np.random.rand(4) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}")

a = np.array([5, 4, 3, 2, 1])
print(
    f"np.array([5, 4, 3, 2, 1]) : a = {a}, a shape = {a.shape}, a datatype = {a.dtype}"
)
