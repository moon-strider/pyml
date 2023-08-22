import numpy as np


class Module:
    def __init__(self) -> None:
        pass


def calculate_padding_1d(
    vector_length: int, kernel_size: int, stride: int
) -> tuple[int, int]:
    if kernel_size > vector_length:
        raise ValueError(
            "Cannot calculate padding for a vector that is smaller than kernel size."
        )

    #left_kernel_part = kernel_size // 2 - (not kernel_size % 2)
    # < or <=?
    total_padding = ((vector_length // stride * stride) + stride - vector_length) if stride <= kernel_size else 0
        #(kernel_size + stride - vector_length % (kernel_size + stride)) % kernel_size
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return (left_padding, right_padding)


def is_symmetrical_1d(vector_length: int) -> bool:
    return vector_length % 2


def ensure_1d(x: np.ndarray) -> np.ndarray:
    """
    Ensure that the input np.ndarray is a vector.

    :param np.ndarray x: the numpy array to check
    :return: the original array
    :rtype: np.ndarray
    :raises ValueError: if the x is not a vector
    """
    if x.ndim != 1:
        raise ValueError("Input data should be a vector.")
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
        raise ValueError("Input data should be a vector or a matrix.")
