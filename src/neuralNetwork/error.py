import numpy as np
from abc import ABC, abstractmethod

class Error(ABC):
    """
    Classe astratta che rappresenta un generico errore.
    """
    
    @abstractmethod
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        raise NotImplementedError()


class MeanSquaredError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.square(target_output - output_nn)
        error_total = np.mean(error_vector)

        return error_total

class MeanAbsoluteError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.abs(target_output - output_nn)
        error_total = np.mean(error_vector)

        return error_total

class MeanAbsolutePercentageError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.abs((target_output - output_nn) / target_output)
        error_total = 100*np.mean(error_vector)

        return error_total


class SquaredError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.sum(error_vector)
        return error_total

class MeanEuclideanError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.linalg.norm(output_nn - target_output, axis=1) #Calcola norma 2 per ogni pattern p
        error_total = np.mean(error_vector)

        return error_total