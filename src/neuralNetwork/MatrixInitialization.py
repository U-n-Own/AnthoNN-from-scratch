import numpy as np
from abc import ABC, abstractmethod


class MatrixInitialization(ABC):
    """
    Classe astratta che rappresenta una generica inizializzazione di una matrice.
    """
    def generate(self, weights_shape: tuple) -> np.matrix:
        raise NotImplementedError()


class ReandomInitialization(MatrixInitialization):

    def __init__(self, min_value_weight: float, max_value_weight: float):

        if min_value_weight > max_value_weight:
            raise ValueError(f"min_value_weight must be ({min_value_weight}) <= max_value_weight ({min_value_weight})")

        self.min_value_weight = min_value_weight
        self.max_value_weight = max_value_weight

    def generate(self, weights_shape: tuple) -> np.matrix:
        return np.matrix(np.random.uniform(low=self.min_value_weight, high=self.max_value_weight, size=weights_shape))

class XavierInitialization(MatrixInitialization):
    def generate(self, weights_shape: tuple) -> np.matrix:
        fan_in, fan_out = weights_shape[0], weights_shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.matrix(np.random.uniform(-limit, limit, weights_shape))

class HeInitialization(MatrixInitialization):
    def generate(self, weights_shape) -> np.matrix:
        fan_in = weights_shape[0]
        limit = np.sqrt(2 / fan_in)
        return np.matrix(np.random.normal(0, limit, weights_shape))
