import numpy as np

from core.utils import ensure_2d
from core.utils import Module

from imgproc.utils import get_center_2d, get_center_1d, \
                        get_pixel


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
    

# class AvgPooling2D(Module):
#     def __init__(self, filter_shape: tuple):
#         super().__init__()

#     def forward(self, x):
#         pass

#     def backward(self, grad):
#         pass
    

# class MaxPooling2D(Module):
#     def __init__(self, filter_shape: tuple, stride=0):
#         super().__init__()
#         self.filter_shape = filter_shape
#         if stride == 0:
#             self.stride = filter.shape[0]
#         else:
#             self.stride = stride
#         self.getmax = lambda x: np.max(x)

#     def forward(self, x): # TODO: add stride
#         x = ensure_2d(x)
#         rows, cols = x.shape
#         filter_rows, filter_cols = self.filter_shape
#         y = np.zeros((rows // self.stride))
#         filter_center = get_center_2d(np.zeros(self.filter_shape))
#         for i in range(x.size):
#             row = i // cols
#             col = i % cols
#             #pool_vec = [np.max([get_pixel()])]

            

#     def backward(self, grad):
#         pass


# class Convolutional2D(Module):
#     def __init__(self, filter_shape: tuple, stride=0):
#         super().__init__()
#         self.filter = np.ones(filter_shape, np.float32)
#         if not stride:
#             stride = filter_shape[0]

#     def forward(self, x):
#         x = ensure_2d(x)
#         rows, cols = x.shape
#         filter_rows, filter_cols = self.filter_shape
#         y = np.zeros((rows // self.stride))
#         center = get_center_2d(np.zeros(self.filter_shape))
#         for i in range(x.size):
#             row = i // cols
#             col = i % cols
#             for j in range(filter_rows * filter_cols):
#                 filter_row = j // filter_cols
#                 filter_col = j % filter_cols
        
#     def backward(self, grad):
#         pass