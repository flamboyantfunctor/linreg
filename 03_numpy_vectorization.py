import numpy as np

"""
CREATING ARRAYS
"""

# Creating a 'scalar' (0D - Array)
s = np.array(10)
print(f"Scalar: {s}, Shape: {s.shape}")

# Creating 'vectors' (1D - Array)
v1 = np.array([1, 2, 3, 4, 5])
print(f"Vector1: {v1}, Shape: {v1.shape}")

v2 = np.array([6, 7, 8, 9, 10])
print(f"Vector2: {v2}, Shape: {v2.shape}")

v3 = np.arange(100)
print(f"Vector3: {v3}, Shape: {v3.shape}")


# Operations on arrays
v1_mul_s = v1 * s
print(f"multiplied: {v1_mul_s}, Shape: {v1_mul_s.shape}")

v1_dot_v2 = np.dot(v1, v2)
print(f"dot product: {v1_dot_v2}, Shape of dotproduct: {v1_dot_v2.shape}")

"""
INDEXING ARRAYS
"""

v4 = v3[:50]
v5 = v3[50:]
print(f"{v4} Shape:{v4.shape} \n {v5} Shape:{v5.shape}")
