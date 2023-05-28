"""

This library will contain our model selection and assessment algorithms.

"""
import copy
import itertools
import numpy as np
import multiprocessing
from src.neuralNetwork.NeuralNetwork import NeuralNetwork


def retrain(self, best_model, best_parameters, target_inputs, target_outputs):
    """ Final retrain the best network after cross-validation process """

    epochs, learning_rate, momentum_term, regularization_term = best_parameters
    best_model.train(target_inputs=target_inputs, target_outputs=target_outputs, epochs=epochs,
                     learning_rate=learning_rate,
                     momentum_term=momentum_term, regularization_term=regularization_term)


def _k_fold_partitioning(target_inputs: np.matrix, target_outputs: np.matrix, k: int, fold_index: int) -> dict:
    """
    K-fold partitioning algorithm

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param fold_index: index of the fold
    :return: model_selection inputs, model_selection outputs, validation inputs, validation outputs
    """

    dataset_dict = {}

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

    dataset_dict['validation_inputs'] = target_inputs[start:end]
    dataset_dict['validation_outputs'] = target_outputs[start:end]

    dataset_dict['training_inputs'] = np.concatenate((target_inputs[:start], target_inputs[end:]))
    dataset_dict['training_outputs'] = np.concatenate((target_outputs[:start], target_outputs[end:]))

    return dataset_dict


def cross_validation(target_inputs: np.matrix, target_outputs: np.matrix, k: int,
                      model: NeuralNetwork, learning_rate, momentum_term, regularization_term, epochs) -> (np.matrix, np.matrix):
    """
    Cross validation algorithm using k folds

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param model: model to use
    return: error_history and validation_error_history for each fold
    """

    training_error_list = []
    validation_error_list = []

    for i in range(k):
        print("Fold ", i)
        modelCopy = copy.deepcopy(model)

        dataset_dict = _k_fold_partitioning(target_inputs, target_outputs, k, fold_index=i)
        training_inputs, training_outputs = dataset_dict['training_inputs'], dataset_dict['training_outputs']
        validation_inputs, validation_outputs = dataset_dict['validation_inputs'], dataset_dict['validation_outputs']
        training_error_history_fold, validation_error_history_fold = modelCopy.train(
            target_inputs_training=training_inputs, target_outputs_training=training_outputs,
            target_inputs_validation=validation_inputs, target_outputs_validation=validation_outputs,
            epochs=epochs, learning_rate=learning_rate, momentum_term=momentum_term,
            regularization_term=regularization_term)

        training_error_list.append(training_error_history_fold)
        validation_error_list.append(validation_error_history_fold)

    return np.matrix(training_error_list), np.matrix(validation_error_list)