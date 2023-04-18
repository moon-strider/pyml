import layer

import numpy as np


linear = layer.linear_layer(3, 2, 0.002)
sigm = layer.sigmoid()
print(f"w: {linear.w}")
print(f"b: {linear.b}")

x = np.random.randn(3, 2)

print(f"x: {x}")
print(f"linear forward: {linear.forward(x)}")
print(f"sigmoid forward: {sigm.forward(x)}")