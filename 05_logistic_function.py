import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Define the inputs and starting parameters
x = np.arange(-10, 10, 0.1)
W_MIN = 0.0
W_MAX = 20.0
W_INIT = 10.0

B_MIN = -10.0
B_MAX = 10.0
B_INIT = 0.0


# Define the logistic function
def logistic_function(x: np.ndarray, w: float, b: float):

    m = x.shape[0]
    y = np.empty(x.shape)

    for i in range(m):
        f_wb = w * x[i] + b
        e_term = pow(np.e, -f_wb)
        y[i] = 1 / (1 + e_term)

    return y


# Compute the y values
y = logistic_function(x, W_INIT, B_INIT)

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)
(line,) = ax.plot(x, y)


ax_w = fig.add_axes([0.25, 0.1, 0.65, 0.03])
w_slider = Slider(ax=ax_w, label="w", valmin=W_MIN, valmax=W_MAX, valinit=W_INIT)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=ax_b,
    label="b",
    valmin=B_MIN,
    valmax=B_MAX,
    valinit=B_INIT,
    orientation="vertical",
)


def update(val):
    line.set_ydata(logistic_function(x, w_slider.val, b_slider.val))
    fig.canvas.draw_idle()


def reset(event):
    w_slider.reset()
    b_slider.reset()


# Connect the update function to sliders change events
w_slider.on_changed(update)
b_slider.on_changed(update)

# Create and place a reset button
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="hotpink")

button.on_clicked(reset)

plt.show()
