import numpy as np

import core.layers as layers
import core.activations as activations

from PIL import Image

from imgproc.utils import is_grey_scale


img = Image.open("img.jpg")
grey = Image.open("gray.jpeg")

print(is_grey_scale(img))
print(is_grey_scale(grey))

linear = layers.Linear(3, 2, 0.002)
sigm = activations.Sigmoid()
print(f"w: {linear.w}")
print(f"b: {linear.b}")

x = np.random.randn(3)

print(f"x: {x}")
print(f"linear forward: {linear.forward(x)}")
print(f"sigmoid forward: {sigm.forward(x)}")

for i in np.random.randn(3, 4):
    print(i)