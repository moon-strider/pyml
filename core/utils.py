import numpy as np


class Module():
    def __init__(self):
        pass


def ensure_1d(x: np.ndarray) -> np.ndarray:
    """
    Ensure that the input np.ndarray is a vector.

    :param np.ndarray x: the numpy array to check
    :return: the original array
    :rtype: np.ndarray
    :raises ValueError: if the x is not a vector
    """
    if x.ndim != 1:
        raise ValueError('Input data should be a vector.')
    return x


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """
    Ensure that the input np.ndarray is a matrix.

    :param np.ndarray x: the numpy array to check
    :return: the original or (if it is possible) reshaped 1d->2d array
    :rtype: np.ndarray
    :raises ValueError: if the x is not a matrix and cannot be reshaped into one
    """
    if x.ndim == 1:
        np.reshape(x, (1, -1))
        return x
    elif x.ndim != 2:
        raise ValueError('Input data should be a vector or a matrix.')