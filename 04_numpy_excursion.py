from scipy.datasets import face
import matplotlib.pyplot as plt

img = face()

plt.imshow(img)
plt.show()

print(img.shape)
print(img.ndim)
print(img[:, :, 0])
