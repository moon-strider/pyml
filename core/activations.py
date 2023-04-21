import numpy as np

from core.utils import ensure_2d
from core.utils import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formula(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        x = ensure_2d(x)
        return self.formula(x)

    def backward(self, x):
        return Sigmoid.formula(x) * (1 - Sigmoid.formula(x))


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        x = ensure_2d(x)
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.x < 0] = 0
        return grad_input
    

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, x):
        x = ensure_2d(x)
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        # we use self.y here because it is the result of tanh(x)
        return grad * (1 - self.y ** 2)