import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

anim = None

x = np.arange(-1, 1, 0.1)

# Ominous Target Function - In reality we'll never know this function
y = 1 * (x**1) - 0.5 * (x**2) - 1 * (x**3) + 2
y_str = r"$f(x) = x - 0.5x^2 - x^3 + 2$"

# Create input data that has 3 polynomial features:
# np.c_ is a function that concatenates the specified values
X = np.c_[x**1, x**2, x**3]

# Assign some random weights.. We need 3 weights because we have 3 features
w_init = np.array([0.36, 0.84, -0.73])

# Assign a random bias value
b_init = 0


def compute_cost(X, y, w, b):
    """Computes the Cost (MSE-mean squared error) for given inputs, targets, weights and bias values"""
    m = len(y)
    h = X @ w + b
    error = h - y
    J = 1 / (2 * m) * np.sum(error**2)
    return J


def update_wb(X, y, w, b, alpha):
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        dj_dw += ((X[i] @ w + b) - y[i]) * X[i]
        dj_db += (X[i] @ w + b) - y[i]

    dj_dw /= m
    dj_db /= m

    w -= alpha * dj_dw
    b -= alpha * dj_db
    w1_slider.set_val(w[0])
    w2_slider.set_val(w[1])
    w3_slider.set_val(w[2])
    b_slider.set_val(b)


def fit(event):

    w = np.array([w1_slider.val, w2_slider.val, w3_slider.val])
    b = b_slider.val
    alpha = float(alpha_btns.value_selected)

    update_wb(X, y, w, b, alpha)


# Setup the matplotlib figure
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0.4, bottom=0.15)


scatter = plt.scatter(x, y, marker="x", c="hotpink", label="datapoints")
(line,) = plt.plot(x, X @ w_init + b_init, label="model")

# Setup of UI elements: 4 Sliders, 1 Button

ax_w1 = fig.add_axes([0.05, 0.15, 0.01, 0.63])
w1_slider = Slider(
    ax=ax_w1,
    label="w1",
    valmin=-1,
    valmax=1,
    valinit=w_init[0],
    orientation="vertical",
)

ax_w2 = fig.add_axes([0.1, 0.15, 0.01, 0.63])
w2_slider = Slider(
    ax=ax_w2,
    label="w2",
    valmin=-1,
    valmax=1,
    valinit=w_init[1],
    orientation="vertical",
)

ax_w3 = fig.add_axes([0.15, 0.15, 0.01, 0.63])
w3_slider = Slider(
    ax=ax_w3,
    label="w3",
    valmin=-1,
    valmax=1,
    valinit=w_init[2],
    orientation="vertical",
)

ax_b = fig.add_axes([0.2, 0.15, 0.01, 0.63])
b_slider = Slider(
    ax=ax_b,
    label="b",
    valmin=-3,
    valmax=3,
    valinit=b_init,
    orientation="vertical",
)

# ax_iter = fig.add_axes([0.25, 0.15, 0.01, 0.63])
# iter_slider = Slider(
#     ax=ax_iter,
#     label="iterations",
#     valmin=1000,
#     valmax=10000,
#     valinit=5000,
#     valstep=1000,
#     orientation="vertical",
# )

ax_alpha = fig.add_axes([0.30, 0.15, 0.05, 0.2])
alpha_btns = RadioButtons(ax=ax_alpha, labels=["0.1", "0.01", "0.001"], active=1)


def start_animation(event):
    global anim
    anim = FuncAnimation(fig, fit, frames=10000, interval=10, repeat=False)


def update(event):
    current_w = np.array([w1_slider.val, w2_slider.val, w3_slider.val])
    current_b = b_slider.val

    line.set_data([x], [X @ current_w + current_b])


def reset(event):
    w1_slider.reset()
    w2_slider.reset()
    w3_slider.reset()
    b_slider.reset()
    # iter_slider.reset()


animateax = fig.add_axes([0.4, 0.025, 0.2, 0.04])
animate_btn = Button(animateax, "Animate!")

resetax = fig.add_axes([0.65, 0.025, 0.2, 0.04])
reset_btn = Button(resetax, "Reset")

target_text = plt.text(-3.2, 1, f"Target Function: {y_str}")

w1_slider.on_changed(update)
w2_slider.on_changed(update)
w3_slider.on_changed(update)
b_slider.on_changed(update)

reset_btn.on_clicked(reset)
animate_btn.on_clicked(start_animation)

plt.show()
