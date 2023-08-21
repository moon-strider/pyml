import numpy as np

from bases.utils import ensure_2d, ensure_1d, is_symmetrical_1d, calculate_padding_1d
from bases.utils import Module

from imgproc.utils import get_center_2d, get_center_1d, get_pixel


class Linear(Module):
    """
    A Neural Network layer that performs a Linear transformation.

    :param int in_dim: number of features in the input
    :param int out_dim: number of features in the output
    :param np.float32 lr: learning rate
    """

    def __init__(self, in_dim: int, out_dim: int, lr: np.float32) -> None:
        """Constructor."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.zeros((out_dim, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass for the layer (A linear transformation).

        :param np.ndarray x: an input array to perform forward pass on
        :return: the dot product of the layer's weights and input, plus layer's bias
        :rtype: np.ndarray
        """
        self.x = x
        x = ensure_2d(x)

        return np.dot(self.w, x) + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Calculate a gradient for back propagation.

        :param np.ndarray grad: a gradient from the next layer
        :return: a gradient to pass to ther previous layer
        :rtype: np.ndarray
        """
        in_grad = np.dot(self.w.T, grad)
        w_grad = np.dot(grad, self.x.T)
        b_grad = np.sum(grad, axis=1, keepdims=True)
        self.w -= w_grad * self.lr
        self.b -= b_grad * self.lr

        return in_grad


class AvgPooling1D(Module):
    def __init__(self, kernel_size: int, stride=None, zero_padding=True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.kernel_center = get_center_1d(kernel_size)
        self.zero_padding = zero_padding
        self.cached = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: check if len > kernel_size
        x = ensure_1d(x)
        kernel_symmetrical = is_symmetrical_1d(self.kernel_size)
        pooled = np.array([])
        req_padding = calculate_padding_1d(len(x), self.kernel_size, self.stride)

        print(f"x: {x}, required padding: {req_padding}")
        
        if self.zero_padding: # and req_padding?
            x = np.pad(x, req_padding, mode='constant')

        start = self.kernel_size // 2 - (not kernel_symmetrical)
        end = len(x) - self.kernel_size // 2 + 1

        print(f"padded x: {x}")
        print(f"for x starting at: {start}, ending with: {end}")

        for i in range(start, end, self.stride):
            print(f"stepping: {x[i - self.kernel_size // 2 : i + self.kernel_size // 2 + 1]}")
            pooled = np.append(pooled, np.mean(x[i - self.kernel_size // 2 : i + self.kernel_size // 2 + 1]))

        self.cached = pooled

        return self.cached

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = np.divide(self.kernel_size, np.power(self.cache, -2))

        return grad


# TODO: AvgPooling2D
# TODO: MaxPooling1D
# TODO: MaxPooling2D
# TODO: Convolutional1D
# TODO: Convolutional2D
