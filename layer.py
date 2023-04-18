import numpy as np


class module():
    def __init__(self):
        pass


class linear_layer(module):
    def __init__(self, in_dim: int, out_dim: int, lr: np.float32):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.zeros((out_dim, 1))


    def forward(self, x):
        self.x = x
        if x.ndim == 1:
            np.reshape(x, (1, -1))
        elif x.ndim != 2:
            raise ValueError('Input data should be a 1D or 2D vector.')
        return np.dot(self.w, x) + self.b


    def backward(self, grad):
        in_grad = np.dot(self.w.T, grad)
        w_grad = np.dot(grad, self.x.T)
        b_grad = np.sum(grad, axis=1, keepdims=True)
        self.w -= w_grad * self.lr
        self.b -= b_grad * self.lr
        return in_grad
    

class sigmoid(module):
    def __init__(self):
        pass


    @staticmethod
    def formula(x):
        return 1 / (1 + np.exp(-x))


    def forward(self, x):
        self.x = x
        if x.ndim == 1:
            np.reshape(x, (1, -1))
        elif x.ndim != 2:
            raise ValueError('Input data should be a 1D or 2D vector.')
        return self.formula(x)
    

    def backward(self, x):
        return sigmoid.formula(x) * (1 - sigmoid.formula(x))
