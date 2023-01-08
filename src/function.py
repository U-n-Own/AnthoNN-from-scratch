import numpy as np


class ActivationFunction:

    def output(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()


class IdentityFunction(ActivationFunction):
    def output(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1


class SigmoideFunction(ActivationFunction):
    """
    The universal approximation theorem states that feedforward neural network with a linear output layer
    and at least one hidden layer with any "squashing" activation function such as Sigmoid or Logistic
    can approximate functions from one finite-dimesional space to another with nonzero amount of error
    with enough hidden units.
    Source: Deep Learning Book by Ian Goodfellow, Yoshua Bengio and Aaron Courville
    """

    """ 
    The universal approximation theorem states that feedforward neural network with a linear output layer
    and at least one hidden layer with any "squashing" activation function such as Sigmoid or Logistic
    can approximate functions from one finite-dimesional space to another with nonzero amount of error
    with enough hidden units.
    Source: Deep Learning Book by Ian Goodfellow, Yoshua Bengio and Aaron Courville 
    """

    def output(self, x: float) -> float:
        return 1.0 / (1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        return np.exp(-x) / (1.0 + np.exp(-x))**2

class ReLuFunction(ActivationFunction):
    """
    Rectified Linear Function is usually the default activation function for feedforward neural networks,
    applying the function on the output of a linear transformation yields a non linear transformation.
    Still very close to linear. Used to solve XOR problem for example.
    Source: Deep Learning Book by Ian Goodfellow, Yoshua Bengio and Aaron Courville
    """

    def output(self, x: float) -> float:
        return np.maximum(0, x)

    def derivative(self, x: float) -> float:
        """ Can't derivate ReLu in 0, it's undefined, we simply return 0 """
        return 1 if x > 0 else 0
