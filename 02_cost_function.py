import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Training data Inputs [size]
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])

# Training data Labels [price]
y_train = np.array(
    [
        250,
        300,
        480,
        430,
        630,
        730,
    ]
)

x_normalized = np.interp(x_train, (x_train.min(), x_train.max()), (-1, 1))
y_normalized = np.interp(y_train, (y_train.min(), y_train.max()), (-1, 1))

WB_MIN = -5
WB_MAX = 5
# Ranges for w, b values
w_range = np.linspace(WB_MIN, WB_MAX, 100)
b_range = np.linspace(WB_MIN, WB_MAX, 100)

# Initial parameter w, b
w_init = -4
b_init = -4

W, B = np.meshgrid(w_range, b_range)


def compute_model(x, w, b):
    """Calculates the prediction values and represents the model"""
    f = np.dot(x, w) + b
    return f


def compute_cost(w, b):
    """Computes the cost for given parameters"""
    m = len(x_normalized)

    total_cost = 0
    for i in range(m):
        f_wb = w * x_normalized[i] + b
        loss = (f_wb - y_normalized[i]) ** 2
        total_cost = total_cost + loss
    total_cost = (1 / (2 * m)) * total_cost
    return total_cost


# Compute the initial linear regression model
f = compute_model(x_normalized, w_init, b_init)

# Compute the the costs
costs = np.vectorize(compute_cost)(W, B)


#### PLOTTING ####
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

# Plotting the linear model
ax1.scatter(x_normalized, y_normalized, marker="x")
(line,) = ax1.plot(x_normalized, compute_model(x_normalized, w_init, b_init))
# Plotting the cost surface
ax2.plot_surface(W, B, costs, cmap="magma", alpha=0.5)
(point,) = ax2.plot(w_init, b_init, compute_cost(w_init, b_init), marker="o", c="b")


# Configure the axes
ax1.set_xlabel("Siz norm")
ax1.set_ylabel("Price norm")
ax1.set_title("Linear Model Plot")

ax2.set_xlabel("Weight")
ax2.set_ylabel("Bias")
ax2.set_zlabel("Cost")
ax2.set_title("Cost Function Plot")


ax_w = fig.add_axes([0.25, 0.1, 0.65, 0.03])
w_slider = Slider(ax=ax_w, label="w", valmin=WB_MIN, valmax=WB_MAX, valinit=w_init)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=ax_b,
    label="b",
    valmin=WB_MIN,
    valmax=WB_MAX,
    valinit=b_init,
    orientation="vertical",
)

# Create and place a reset button
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset")


# Define an update function for the sliders on-changed event
def update(val):
    """Updates the y values according to the w and b slider values"""
    line.set_data(
        [x_normalized], [compute_model(x_normalized, w_slider.val, b_slider.val)]
    )
    point.set_data(
        [w_slider.val],
        [b_slider.val],
    )
    point.set_3d_properties([compute_cost(w_slider.val, b_slider.val)])

    fig.canvas.draw_idle()


# Define a reset function for the buttons on-clicked event
def reset(event):
    """Resets the slider values to the initial values of w and b"""
    w_slider.reset()
    b_slider.reset()


# Connect the update function to sliders on-changed events
w_slider.on_changed(update)
b_slider.on_changed(update)

# Connect the onclick event of the reset button to reset function
button.on_clicked(reset)

plt.show()
