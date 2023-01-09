""" 

This library will contain our model selection and assessment algorithms.

"""

import numpy as np


def _k_fold_partitioning(target_inputs: np.matrix, target_outputs: np.matrix, k: int, fold_index: int) -> (np.matrix, np.matrix, np.matrix, np.matrix):
    """
    K-fold partitioning algorithm

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param fold_index: index of the fold
    :return: training inputs, training outputs, validation inputs, validation outputs
    """
    
    len_fold = target_inputs.shape[0] // k
    start = len_fold * fold_index
    end = start + len_fold
    
    validation_inputs = target_inputs[start:end]
    validation_outputs = target_outputs[start:end]
    
    training_inputs = np.concatenate((target_inputs[:start], target_inputs[end:]))
    training_outputs = np.concatenate((target_outputs[:start], target_outputs[end:]))
    
    return training_inputs, training_outputs, validation_inputs, validation_outputs

def _cross_validation(target_inputs: np.matrix, target_outputs: np.matrix, k: int, model, parameters_set) -> float:
    """
    Cross validation algorithm using k folds 

    :param target_inputs: matrix containing the inputs
    :param target_outputs: matrix containing the outputs
    :param k: number of folds
    :param model: model to use
    :param kwargs: optional parameters for the model
    :return: the average error
    """
    
    error = 0
    for i in range(k):
        training_inputs, training_outputs, validation_inputs, validation_outputs = k_fold_partitioning(target_inputs, target_outputs, k, i)
        epochs, learning_rate, momentum_term, regularization_term = parameters_set
        model.train(training_inputs, training_outputs, epochs, learning_rate, momentum_term, regularization_term)
        #TODO Check implementation of validate error function
        error += model.validate(validation_inputs, validation_outputs)
    
    return error / k
    
def grid_search(model, parameters_grid: dict, **kwargs) -> dict:
    """
    Grid search algorithm, GS has complexity O(n^d) 
    where n is the number of parameters and d the number of values for each parameter.
    Without counting the 

    :param model: model to use
    :param parameters: dictionary of parameters to test
    
    parameters_grid_example = {
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum_term': [0.1, 0.3, 0.9],
        'regularization_term': [0.1, 0.01, 0.001],
        'epochs': [100, 200, 300]
    }
    
    :param kwargs: optional parameters for the future hyperparameters
    :return: the best parameters
    """
    
    best_parameters = None
    best_error = float('inf')
    for parameters_set in parameters_grid:
        error = cross_validation(target_inputs, target_outputs, k, model, **parameters_set)
        if error < best_error:
            best_error = error
            best_parameters = parameters_set
            
    return best_parameters
