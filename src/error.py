import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


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
        error_vector = np.square(target_output - output_nn)
        error_total = np.mean(error_vector)

        assert error_total == mean_squared_error(target_output, output_nn)

        return error_total

class MeanAbsoluteError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.abs(target_output - output_nn)
        error_total = np.mean(error_vector)

        assert error_total == mean_absolute_error(target_output, output_nn)

        return error_total

class MeanAbsolutePercentageError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.abs((target_output - output_nn) / target_output)
        error_total = np.mean(error_vector)

        assert round(error_total, 10) == round(mean_absolute_percentage_error(target_output, output_nn), 10)

        return error_total


class SquaredError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.sum(error_vector)
        return error_total

class MeanEuclidianError(Error):
    def calculate_total_error(self, target_output: np.matrix, output_nn: np.matrix) -> np.float64:
        # TODO controllare se è corretto

        error_vector = np.sum(np.square(target_output - output_nn), axis=1)
        error_total = np.mean(np.sqrt(error_vector))


        return error_total