import numpy as np

from core.utils import ensure_2d
from core.utils import Module

from imgproc.utils import get_center_2d, get_center_1d, \
                        get_pixel


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
    def __init__(self, filter_shape: tuple, stride=0):
        super().__init__()
        self.filter_shape = filter_shape
        if stride == 0:
            self.stride = filter.shape[0]
        else:
            self.stride = stride
        self.getmax = lambda x: np.max(x)

    def forward(self, x): # TODO: add stride
        x = ensure_2d(x)
        rows, cols = x.shape
        filter_rows, filter_cols = self.filter_shape
        y = np.zeros((rows // self.stride))
        filter_center = get_center_2d(np.zeros(self.filter_shape))
        for i in range(x.size):
            row = i // cols
            col = i % cols
            #pool_vec = [np.max([get_pixel()])]

            

    def backward(self, grad):
        pass


class Convolutional2D(Module):
    def __init__(self, filter_shape: tuple, stride=0):
        super().__init__()
        self.filter = np.ones(filter_shape, np.float32)
        if not stride:
            stride = filter_shape[0]

    def forward(self, x):
        x = ensure_2d(x)
        rows, cols = x.shape
        filter_rows, filter_cols = self.filter_shape
        y = np.zeros((rows // self.stride))
        center = get_center_2d(np.zeros(self.filter_shape))
        for i in range(x.size):
            row = i // cols
            col = i % cols
            for j in range(filter_rows * filter_cols):
                filter_row = j // filter_cols
                filter_col = j % filter_cols
        
    def backward(self, grad):
        pass