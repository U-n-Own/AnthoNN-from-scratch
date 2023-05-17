"""

This library will contain our model selection and assessment algorithms.

"""
import copy

import numpy as np
from matplotlib import pyplot as plt

from src.neuralNetwork.NeuralNetwork import NeuralNetwork
from src.training.crossvalidation import _k_fold_partitioning


def biasVariance_crossValidation(target_inputs: np.matrix, target_outputs: np.matrix, k: int,
                                 model: NeuralNetwork, learning_rate, momentum_term, regularization_term, epochs):
    """
    Cross validation algorithm using k folds. This function is used to compute the bias and variance of the model

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param model: model to use
    :param kwargs: optional parameters for the model
    :return: error_history and validation_error_history for each fold
    """

    training_error_list = []
    validation_error_list = []

    for i in range(k):
        print("Fold ", i)


        training_inputs, training_outputs, \
            validation_inputs, validation_outputs = _k_fold_partitioning(target_inputs, target_outputs, k, i)

        modelCopy = copy.deepcopy(model)
        training_error_history, validation_error_history = modelCopy.train(
            target_inputs_training = training_inputs, target_outputs_training = training_outputs,
            target_inputs_validation=validation_inputs, target_outputs_validation=validation_outputs,
            epochs=epochs, learning_rate=learning_rate, momentum_term=momentum_term,
            regularization_term=regularization_term)

        training_error_list.append(training_error_history)
        validation_error_list.append(validation_error_history)

    return np.matrix(training_error_list), np.matrix(validation_error_list)


# Qui sotto ci sono le funzioni per fare il plot della varianza e del bias

def plot_training(training_error_list):
    mean_training = np.mean(training_error_list, axis=0)
    std_training = np.std(training_error_list, axis=0)


    #Training
    for training_error in training_error_list:
        training_plot, = plt.plot(training_error.tolist()[0], color = 'lightblue', label='training')
    mean_training_plot, = plt.plot(mean_training.tolist()[0], color = 'blue', label='mean on training')
    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_training-std_training)[0],
                     np.array(mean_training+std_training)[0],alpha=.1, color = 'lightblue')

    return training_plot, mean_training_plot

def plot_validation(validation_error_list):
    mean_validation = np.mean(validation_error_list, axis=0)
    std_validation = np.std(validation_error_list, axis=0)

    #Validation
    for validation_error in validation_error_list:
        validation_plot, = plt.plot(validation_error.tolist()[0], color = 'pink', label='validation')
    mean_validation_plot, = plt.plot(mean_validation.tolist()[0], color = 'red', label='mean on validation')

    plt.fill_between(range(std_validation.shape[1]),
                     np.array(mean_validation-std_validation)[0],
                     np.array(mean_validation+std_validation)[0],alpha=.1, color = 'pink')
    return validation_plot, mean_validation_plot


def plot_mean_and_std_validation(training_error_list, validation_error_list):
    mean_training = np.mean(training_error_list, axis=0)
    std_training = np.std(training_error_list, axis=0)
    mean_validation = np.mean(validation_error_list, axis=0)
    std_validation = np.std(validation_error_list, axis=0)

    plt.plot(mean_training.tolist()[0], color='blue', label='mean on training')
    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_training - std_training)[0],
                     np.array(mean_training + std_training)[0], alpha=.1, color='lightblue')

    # Validation
    plt.plot(mean_validation.tolist()[0], color='red', label='mean on validation')

    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_validation - std_validation)[0],
                     np.array(mean_validation + std_validation)[0], alpha=.1, color='pink')
