import numpy as np

from bases.utils import calculate_padding_1d


def test_padding():
    for i in range(3, 20):
        assert not (i + np.sum(calculate_padding_1d(i, 3, 3))) % 3

    