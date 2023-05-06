"""

This library will contain our model selection and assessment algorithms.

"""
import math
import itertools
import numpy as np
import multiprocessing

from src.neuralNetwork.NeuralNetwork import NeuralNetwork

def retrain(self, best_model, best_parameters, target_inputs, target_outputs):
    """ Final retrain the best network after cross-validation process """

    epochs, learning_rate, momentum_term, regularization_term = best_parameters
    best_model.train(target_inputs=target_inputs, target_outputs=target_outputs, epochs=epochs, learning_rate=learning_rate,
                momentum_term = momentum_term, regularization_term=regularization_term)

def _k_fold_partitioning(target_inputs: np.matrix, target_outputs: np.matrix, k: int, fold_index: int) -> (
np.matrix, np.matrix, np.matrix, np.matrix):
    """
    K-fold partitioning algorithm

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param fold_index: index of the fold
    :return: model_selection inputs, model_selection outputs, validation inputs, validation outputs
    """

    if k < 1:
        raise ValueError(f"k ({k}) must be greater than 0")
    if k > target_inputs.shape[0]:
        raise ValueError(f"k ({k}) must be less than the number of samples ({target_inputs.shape[0]})")

    len_fold = target_inputs.shape[0] // k
    start = len_fold * fold_index
    end = start + len_fold

    if start >= target_inputs.shape[0]:
        raise ValueError(f"index out of range (start = {start} >= target_inputs.shape[0] = {target_inputs.shape[0]})")
    if start < 0:
        raise ValueError(f"index out of range (start = {start} < 0)")

    validation_inputs = target_inputs[start:end]
    validation_outputs = target_outputs[start:end]

    training_inputs = np.concatenate((target_inputs[:start], target_inputs[end:]))
    training_outputs = np.concatenate((target_outputs[:start], target_outputs[end:]))

    return training_inputs, training_outputs, validation_inputs, validation_outputs


def _cross_validation(target_inputs: np.matrix, target_outputs: np.matrix, k: int,
                      model: NeuralNetwork, parameters_set, error_queue):
    """
    Cross validation algorithm using k folds

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param model: model to use
    :return: the average error
    """

    learning_rate, momentum_term, regularization_term, epochs = parameters_set

    training_error_history = np.zeros(epochs)
    validation_error_history = np.zeros(epochs)

    for i in range(k):
        print("Fold ", i)
        training_inputs, training_outputs, validation_inputs, validation_outputs = _k_fold_partitioning(target_inputs,
                                                                                                       target_outputs,
                                                                                                       k, i)

        training_error_history_fold, validation_error_history_fold = model.train(
            target_inputs_training = training_inputs, target_outputs_training = training_outputs,
            target_inputs_validation=validation_inputs, target_outputs_validation=validation_outputs,
            epochs=epochs, learning_rate=learning_rate, momentum_term=momentum_term,
            regularization_term=regularization_term)

        training_error_history += training_error_history_fold
        validation_error_history += validation_error_history_fold

    parameters_set = (epochs, learning_rate, momentum_term, regularization_term)
    error_queue.put((training_error_history/k, validation_error_history/k, parameters_set))


def grid_search(parameters_grid: dict, target_inputs: np.matrix, target_outputs: np.matrix, k: int = 5):
    """
    Grid search algorithm, GS has complexity O(n^d)
    where n is the number of parameters and d the number of values for each parameter.
    Without counting the

    :param target_inputs: matrice di input dove verrà eseguito il k-fold
    :param target_outputs: matrice di output dove verrà eseguito il k-fold
    :param parameters_grid: dictionary of parameters to test
    :param k: number of folds

    parameters_grid_example = {
        'model': [NeuralNetwork], # model to use
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum_term': [0.1, 0.3, 0.9],
        'regularization_term': [0.1, 0.01, 0.001],
        'epochs': [100, 200, 300]
    }

    :return: the best parameters
    """
    best_parameters = None
    best_error = None

    error_queue = multiprocessing.Queue()
    process_list = []

    # Remove the model from the parameters grid and create a dictionary with only models
    models = parameters_grid.pop('model')
    # Combination of hyperparameters excluding the model
    combination_grid = list(itertools.product(*parameters_grid.values()))

    for model in models:
        for parameters_set in combination_grid:
            process = multiprocessing.Process(target=_cross_validation, args=(target_inputs, target_outputs, k, model, parameters_set, error_queue))
            process_list.append(process)

    for process in process_list:
        process.start()
        print("Process {} started".format(process.pid))
    for process in process_list:
        process.join()
        print("Process ", process.pid, " terminated")

    # error_queue has a tuple of (error, parameters_set)
    # extract best error and best_parameters
    parameters_list = []
    training_error_history_list = []
    validation_error_history_list = []
    best_validation_error = None

    while not error_queue.empty():
        error = error_queue.get()
        training_error_history_list.append(error[0])
        validation_error_history_list.append(error[1])
        parameters_list.append(error[2])

        if best_error is None or error[0][-1] < best_error:
            best_validation_error = error[0][-1]
            best_parameters = error[2]

    return {
        'best_error': best_error,
        'best_parameters': best_parameters,
        'training_error_history_list': training_error_history_list,
        'validation_error_history_list': validation_error_history_list,
        'best_validation_error': best_validation_error,
        'parameters_list': parameters_list
    }