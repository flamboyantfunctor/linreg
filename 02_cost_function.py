import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

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

# Initial parameter w, b
w_init = 100
b_init = 0

# Ranges for w, b values
w_range = np.linspace(50, 500, 100)
b_range = np.linspace(-100, 450, 100)


def compute_model(x, w, b):
    """Calculates the prediction values and represents the model"""
    f = np.dot(x, w) + b
    return f


def compute_cost(x, y, w, b):
    """Computes the cost for given parameters"""
    m = len(x)

    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        loss = (f_wb - y[i]) ** 2
        total_cost = total_cost + loss
    total_cost = (1 / (2 * m)) * total_cost
    return total_cost


# Compute the linear regression model
f = compute_model(x_train, w_init, b_init)

# Compute the the costs
costs = [compute_cost(x_train, y_train, w, b_init) for w in w_range]

# Interpolate a smooth spline for plotting
spline = make_interp_spline(w_range, costs, k=2)
w_smoothed = spline(w_range)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
fig.subplots_adjust(left=0.25, bottom=0.25)
(line,) = ax1.plot(x_train, f)
(para,) = ax2.plot(w_range, w_smoothed)
(point,) = ax2.plot(
    w_init,
    compute_cost(x_train, y_train, w_init, b_init),
    marker="o",
)

ax1.scatter(x_train, y_train, marker="x", c="r", label="actual values")
ax1.set_xlabel("Size [1000 sqft]")
ax1.set_ylabel("Price [1000 $]")


ax_w = fig.add_axes([0.25, 0.1, 0.65, 0.03])
w_slider = Slider(
    ax=ax_w, label="w", valmin=w_range.min(), valmax=w_range.max(), valinit=w_init
)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=ax_b,
    label="b",
    valmin=b_range.min(),
    valmax=b_range.max(),
    valinit=b_init,
    orientation="vertical",
)


# Define an update function
def update(val):
    line.set_ydata(compute_model(x_train, w_slider.val, b_slider.val))
    point.set_xdata(w_slider.val)
    point.set_ydata(compute_cost(x_train, y_train, w_slider.val, b_init))
    fig.canvas.draw_idle()


# Connect the update function to sliders change events
w_slider.on_changed(update)
b_slider.on_changed(update)

# Create and place a reset button
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="hotpink")


# Define a reset event
def reset(event):
    w_slider.reset()
    b_slider.reset()


# Listen to click events on the button
button.on_clicked(reset)

# Show the figure
plt.show()
