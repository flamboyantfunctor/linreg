import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Define the x inputs
x = np.arange(-1, 1, 0.01)

# Define initial parameters w, b
W_MIN = -50.0
W_MAX = 50.0
W_INIT = 10.0

B_MIN = -20.0
B_MAX = 20.0
B_INIT = 0.0

COLOR = "forestgreen"


# Define the sigmoid function for logistic regression
def sigmoid(x: np.ndarray, w: float, b: float) -> np.ndarray:
    z = np.dot(x, w) + b
    y = 1 / (1 + np.exp(-z))
    return y


# Compute the y values
y = sigmoid(x, W_INIT, B_INIT)

# Create the plot, adjust it's position and plot the graph
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)
(line,) = ax.plot(x, y, c=COLOR)

# Create two sliders for the w and b parameter
ax_w = fig.add_axes([0.25, 0.1, 0.65, 0.03])
w_slider = Slider(
    ax=ax_w, label="w", color=COLOR, valmin=W_MIN, valmax=W_MAX, valinit=W_INIT
)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=ax_b,
    label="b",
    color=COLOR,
    valmin=B_MIN,
    valmax=B_MAX,
    valinit=B_INIT,
    orientation="vertical",
)

# Create and place a reset button
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor=COLOR)


# Define an update function for the sliders on-changed event
def update(val):
    """Updates the y values according to the w and b slider values"""
    line.set_ydata(sigmoid(x, w_slider.val, b_slider.val))
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

# Show the plot
plt.show()
