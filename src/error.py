import numpy as np


class Error:

    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        if target_output is None:
            raise ValueError("target_output must be != None")
        if output_nn is None:
            raise ValueError("output_nn must be != None")
        if target_output.shape != output_nn.shape:
            raise ValueError(
                f"target_output ({target_output.shape}) and output_nn ({output_nn.shape}) must have the same shape")

        raise NotImplementedError()


class MeanSquaredError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.mean(error_vector)
        return error_total

class MeanAbsoluteError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.abs(target_output - output_nn), axis=1)
        error_total = np.mean(error_vector)
        return error_total


class SquaredError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.sum(error_vector)
        return error_total

class MeanEuclidianError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.mean(np.sqrt(error_vector))
        return error_total