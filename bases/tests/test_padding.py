import numpy as np

from bases.utils import calculate_padding_1d


def test_padding():
    input_length, stride, kernel_size = 255, 77, 2
    assert np.sum(calculate_padding_1d(input_length, kernel_size, stride)) == 0
    input_length, stride, kernel_size = 255, 2, 77
    assert np.sum(calculate_padding_1d(input_length, kernel_size, stride)) == 1