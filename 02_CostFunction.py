import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

from matplotlib.widgets import Slider, Button

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
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

w_init = 100
b_init = 50

w_range = np.arange(start=0, stop=450, step=50)


def compute_model(x, w, b):
    """Calculates the prediction values and represents the model"""
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = x[i] * w + b
    return f_wb


def compute_cost(x: list, y: list, w: list, b: int) -> list:
    """Computes the cost"""
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


tmp_f_wb = compute_model(x_train, w_init, b_init)
costs = [compute_cost(x_train, y_train, w, b_init) for w in w_range]

w_new = np.linspace(0, 450, 50)
spl = make_interp_spline(w_range, costs, k=2)
w_smoothed = spl(w_new)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
fig.subplots_adjust(left=0.25, bottom=0.25)
(line,) = ax1.plot(x_train, tmp_f_wb)
(para,) = ax2.plot(w_new, w_smoothed)

ax1.scatter(x_train, y_train, marker="x", c="r", label="actual values")


ax_color = "hotpink"
ax1.set_xlabel("Size [1000 sqft]")
ax1.set_ylabel("Price [1000 $]")


ax_w = fig.add_axes([0.25, 0.1, 0.65, 0.03])
w_slider = Slider(ax=ax_w, label="w", valmin=0, valmax=300, valinit=w_init)

ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=ax_b, label="b", valmin=-200, valmax=300, valinit=b_init, orientation="vertical"
)


def update(val):
    line.set_ydata(compute_model(x_train, w_slider.val, b_slider.val))
    fig.canvas.draw_idle()


w_slider.on_changed(update)
b_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="hotpink")


def reset(event):
    w_slider.reset()
    b_slider.reset()


button.on_clicked(reset)

plt.show()
