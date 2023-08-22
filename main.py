import numpy as np

import bases.layers as layers
import bases.activations as activations
from bases.utils import calculate_padding_1d

for kernel_size in range(2, 500):
    for input_len in range(kernel_size, 500):
        for stride in range(1, 100):
            print("-----------------------------------------------------------")
            print(np.sum(calculate_padding_1d(input_len, kernel_size, stride)))
            print(f"in: {input_len}, kernel: {kernel_size}, stride: {stride}")
            print("-----------------------------------------------------------")