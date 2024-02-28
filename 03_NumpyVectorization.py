import numpy as np

"""
CREATING ARRAYS
"""

# Creating a 'scalar' (0D - Array)
s = np.array(10)
print(f"Scalar: {s}, Shape of Scalar: {s.shape}")

# Creating 'vectors' (1D - Array)
v1 = np.array([1, 2, 3, 4, 5])
print(f"Vector: {v1}, Shape of Vector: {v1.shape}")

v2 = np.array([6, 7, 8, 9, 10])
print(f"Vector: {v2}, Shape of Vector: {v2.shape}")

# Operations on arrays
v1_mul_s = v1 * s
print(f"multiplied: {v1_mul_s}, Shape of Vector: {v1_mul_s.shape}")

v1_dot_v2 = np.dot(v1, v2)
print(f"dot product: {v1_dot_v2}, Shape of dotproduct: {v1_dot_v2.shape}")

"""
INDEXING ARRAYS
"""
