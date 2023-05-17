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


class SigmoidFunction(ActivationFunction):
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

class LeakReLuFunction(ActivationFunction):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def output(self, x: float) -> float:
        return np.maximum(self.alpha * x, x)

    def derivative(self, x: float) -> float:
        return 1 if x > 0 else self.alpha

class TanhFunction(ActivationFunction):
    def output(self, x: float) -> float:
        return np.tanh(x)

    def derivative(self, x: float) -> float:
        return 1 - np.tanh(x)**2

class SoftplusFunction(ActivationFunction):
    def output(self, x: float) -> float:
        return np.log(1+np.exp(x))

    def derivative(self, x: float)-> float:
        return np.exp(x) / (np.exp(x)+1)

class SiluFunction(ActivationFunction):
    def output(self, x: float) -> float:
        return x * self._sigmoid(x)

    def derivative(self, x: float) -> float:
        return self._sigmoid(x) + x * self._sigmoid(x) * (1 - self._sigmoid(x))

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1 + np.exp(-x))

class SELUFunction(ActivationFunction):
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def output(self, x: float) -> float:
        return self.scale * (np.where(x >= 0, x, self.alpha * (np.exp(x) - 1)))

    def derivative(self, x: float) -> float:
        return self.scale * (np.where(x >= 0, 1, self.alpha * np.exp(x)))

class ELUFunction(ActivationFunction):
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("Alpha must be a positive value.")
        self.alpha = alpha

    def output(self, x: float) -> float:
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: float) -> float:
        return np.where(x >= 0, 1, self.alpha * np.exp(x))


