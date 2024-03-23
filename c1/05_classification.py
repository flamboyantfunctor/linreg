import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

x = np.array([1.2, 1.4, 2.3, 2.7, 3.1, 4.0])
y = np.array([0, 0, 0, 1, 1, 1])

w = 0.5
b = -0.8


def linear_model(x: np.ndarray, w: float, b: float) -> np.ndarray:
    y = np.dot(x, w) + b
    return y


def predict(x):
    y = w * x + b
    return 1 if y >= 0.5 else 0


benigns = [(x[i], y[i]) for i in range(len(x)) if y[i] == 0]
malignants = [(x[i], y[i]) for i in range(len(x)) if y[i] == 1]


fig = plt.figure(figsize=(10, 6))
model = plt.plot(x, linear_model(x, w, b))
plt.plot(x, [0.5 for _ in x], linestyle="--", c="black")
benignpoints = plt.scatter(*zip(*benigns), marker="o", c="b", label="benign tumors")
malignantpoints = plt.scatter(
    *zip(*malignants), marker="x", c="r", label="malignant tumors"
)


def onClick(event):
    if event.button is MouseButton.LEFT:
        global x, y
        selected_x = event.xdata
        x = np.append(x, selected_x)
        y = np.append(y, predict(selected_x))
        print(x, y)


plt.connect("button_press_event", onClick)


plt.annotate(
    "Threshold",
    xy=(1.2, 0.5),
    xytext=(1.2, 0.55),
)

plt.title("Tumor Classification")
plt.xlabel(f"tumor size [cm]")
plt.ylabel(f"malignant tumor?")
plt.yticks([0, 0.5, 1])
plt.show()
