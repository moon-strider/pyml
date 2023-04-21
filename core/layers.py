import numpy as np

from core.utils import ensure_2d
from core.utils import Module

from numpy.lib.stride_tricks import as_strided


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, lr: np.float32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.zeros((out_dim, 1))

    def forward(self, x):
        self.x = x
        x = ensure_2d(x)
        return np.dot(self.w, x) + self.b

    def backward(self, grad):
        in_grad = np.dot(self.w.T, grad)
        w_grad = np.dot(grad, self.x.T)
        b_grad = np.sum(grad, axis=1, keepdims=True)
        self.w -= w_grad * self.lr
        self.b -= b_grad * self.lr
        return in_grad
    

class AvgPooling2D(Module):
    def __init__(self, filter_shape: tuple):
        super().__init__()

    def forward(self, x):
        pass

    def backward(self, grad):
        pass
    

class MaxPooling2D(Module):
    def __init__(self, filter_shape: tuple):
        super().__init__()

    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Convolutional2D(Module):
    def __init__(self, filter_shape: tuple, stride=0):
        super().__init__()
        self.filter = np.ones(filter_shape, np.float32)
        if not stride:
            stride = filter_shape[0]

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("An input to a 2D convolution must be a 2D array.")

    def backward(self, grad):
        pass