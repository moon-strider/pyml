import numpy as np


class Module():
    def __init__(self):
        pass


def ensure_1d(x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError('Input data should be a vector.')
    return x


def ensure_2d(x: np.ndarray) -> None:
    if x.ndim == 1:
        np.reshape(x, (1, -1))
        return x
    elif x.ndim != 2:
        raise ValueError('Input data should be a vector or a matrix.')