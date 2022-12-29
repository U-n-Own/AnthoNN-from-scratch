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
    def output(self, x: float) -> float:
        return 1.0 / (1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        return x * (1.0 - x)
