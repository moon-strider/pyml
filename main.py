import core.layers as layers
import core.activations as activations

import numpy as np


linear = layers.Linear(3, 2, 0.002)
sigm = activations.Sigmoid()
print(f"w: {linear.w}")
print(f"b: {linear.b}")

x = np.random.randn(3)

print(f"x: {x}")
print(f"linear forward: {linear.forward(x)}")
print(f"sigmoid forward: {sigm.forward(x)}")