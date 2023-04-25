import numpy as np

from core.utils import ensure_2d
from core.utils import Module


class Sigmoid(Module):
    """
    A sigmoid activation Neural Network layer.
    """
    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

    @staticmethod
    def formula(x: np.ndarray) -> np.ndarray:
        """
        A sigmoid function formula.

        :param np.ndarray x: an input array to pass to the sigmoid function
        :return: the result of applying the sigmoid function to the input array
        :rtype: np.ndarray
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x) -> np.ndarray:
        """
        Performs a forward pass for the layer (A sigmoid activation function).

        :param np.ndarray x: an input array to perform forward pass on
        :return: the squished input array data in the ranges of [0, 1]
        :rtype: np.ndarray
        """
        self.x = x
        x = ensure_2d(x)
        return self.formula(x)

    def backward(self, x) -> np.ndarray:
        """
        Calculate a gradient for back propagation.

        :param np.ndarray grad: a gradient from the next layer
        :return: a gradient to pass to ther previous layer
        :rtype: np.ndarray
        """
        return Sigmoid.formula(x) * (1 - Sigmoid.formula(x))


class ReLU(Module):
    """
    A Rectified Linear Unit activation Neural Network layer.
    """
    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.x = None

    def forward(self, x) -> np.ndarray:
        """
        Performs a forward pass for the layer (A ReLU activation function).

        :param np.ndarray x: an input array to perform forward pass on
        :return: transformed array, so that the x[x < 0] = 0, x[x>=0] = x
        :rtype: np.ndarray
        """
        self.x = x
        x = ensure_2d(x)
        return np.maximum(0, x)

    def backward(self, grad_output) -> np.ndarray:
        """
        Calculate a gradient for back propagation.

        :param np.ndarray grad: a gradient from the next layer
        :return: a gradient to pass to ther previous layer
        :rtype: np.ndarray
        """
        grad_input = grad_output.copy()
        grad_input[self.x < 0] = 0
        return grad_input
    

class Tanh(Module):
    """
    A hyperbolic tangent actication Neural Network layer.
    """
    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.y = None

    def forward(self, x) -> np.ndarray:
        """
        Performs a forward pass for the layer (A tanh activation function).

        :param np.ndarray x: an input array to perform forward pass on
        :return: the squished input array data in the ranges of [-1, 1]
        :rtype: np.ndarray
        """
        x = ensure_2d(x)
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad) -> np.ndarray:
        """
        Calculate a gradient for back propagation.

        :param np.ndarray grad: a gradient from the next layer
        :return: a gradient to pass to ther previous layer
        :rtype: np.ndarray
        """
        return grad * (1 - self.y ** 2)