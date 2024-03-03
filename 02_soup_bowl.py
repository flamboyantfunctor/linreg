import numpy as np
import matplotlib.pyplot as plt

w_range = np.linspace(-20, 20, 100)
b_range = np.linspace(-20, 20, 100)

W, B = np.meshgrid(w_range, b_range)


# Define the cost function
def cost_function(w, b):
    return w**2 + b**2


cost = np.vectorize(cost_function)(W, B)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(W, B, cost, cmap="viridis")

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Cost")
ax.set_title("Cost Function Surface Plot")

plt.show()
