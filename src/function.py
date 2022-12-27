import numpy as np


class IdentityFunction:
    def output(x: float) -> float:
        return x

    def derivative(x: float) -> float:
        return 1


class SigmoideFunction:
    def output(x: float) -> float:
        return 1.0 / (1 + np.exp(-x))

    def derivative(x: float) -> float:
        return x * (1.0 - x)
