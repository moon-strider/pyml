import numpy as np

import core.layers as layers
import core.activations as activations

avg_pooling_1d = layers.AvgPooling1D(3)
to_pool = np.array([1, 2, 3])
to_pool_2 = np.array([1, 2, 3, 4])
to_pool_3 = np.array([1, 2, 3, 4, 5])
to_pool_4 = np.array([1, 2, 3, 4, 5, 6])

res_pool = avg_pooling_1d.forward(to_pool)
res_pool_2 = avg_pooling_1d.forward(to_pool_2)
res_pool_3 = avg_pooling_1d.forward(to_pool_3)
res_pool_4 = avg_pooling_1d.forward(to_pool_4)

print(avg_pooling_1d.stride)

print(res_pool)
print(res_pool_2)
print(res_pool_3)
print(res_pool_4)