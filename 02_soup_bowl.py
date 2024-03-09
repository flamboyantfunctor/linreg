import numpy as np
import matplotlib.pyplot as plt

w_range = np.linspace(-20, 20, 100)
b_range = np.linspace(-20, 20, 100)

W, B = np.meshgrid(w_range, b_range)


# Define the "soup bowl" cost function
def cost_function(w: float, b: float) -> float:
    """Computes the cost for given w and b value"""
    return w**2 + b**2


"""
Let's "vectorize" the cost function with numpy's vectorize function. 
"Vectorizing a function" means converting a function that previously
only operated on scalar values to a function thatcan operate on arrays 
(list of scalar values). Now we dont have to use something like loops
oder maps and get the performance benefits of array operations.
"""
cost_vectorized = np.vectorize(cost_function)
cost = cost_vectorized(W, B)

# Plotting the results with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(W, B, cost, cmap="viridis")

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Cost")
ax.set_title("Cost Function Surface Plot")

plt.show()
