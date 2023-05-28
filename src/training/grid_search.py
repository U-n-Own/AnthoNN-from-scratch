import copy
import itertools
import numpy as np
import multiprocessing
from src.neuralNetwork.NeuralNetwork import NeuralNetwork
from training.crossValidation import cross_validation


def _grid_cross_validation(target_inputs: np.matrix, target_outputs: np.matrix,
                           k: int, model: NeuralNetwork, parameters_set) -> dict:
    
    learning_rate, momentum_term, regularization_term, epochs = parameters_set
    training_error_matrix, validation_error_matrix = cross_validation(target_inputs, target_outputs, k, model, learning_rate, momentum_term, regularization_term, epochs)

    training_error_history = np.mean(training_error_matrix, axis=0)
    validation_error_history = np.mean(validation_error_matrix, axis=0)
    return {'training_error_history': training_error_history,
             'validation_error_history': validation_error_history,
             'parameters_set': parameters_set}

def grid_search(parameters_grid: dict, target_inputs: np.matrix, target_outputs: np.matrix,
                k: int = 5, num_processes: int = 8) -> dict:
    """
    Grid search algorithm, GS has complexity O(n^d)
    where n is the number of parameters and d the number of values for each parameter.
    Without counting the

    :param target_inputs: matrice di input dove verrà eseguito il k-fold
    :param target_outputs: matrice di output dove verrà eseguito il k-fold
    :param parameters_grid: dictionary of parameters to test
    :param k: number of folds
    :param num_processes: number of processes to use

    parameters_grid_example = {
        'model': [NeuralNetwork], # model to use
        'learning_rate': [0.1, 0.01, 0.001],
        'momentum_term': [0.1, 0.3, 0.9],
        'regularization_term': [0.1, 0.01, 0.001],
        'epochs': [100, 200, 300]
    }

    :return: the best parameters
    """

    args_list = []

    # Remove the model from the parameters grid and create a dictionary with only models
    models = parameters_grid.pop('model')
    # Combination of hyperparameters excluding the model
    combination_grid = list(itertools.product(*parameters_grid.values()))
    for model in models:
        for parameters_set in combination_grid:
            args = (target_inputs, target_outputs, k, model, parameters_set)
            args_list.append(args)


    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(_grid_cross_validation, args_list)


    return _create_info_dict(results)

def _create_info_dict(results) -> dict:
    info_dict = {
        'parameters_list': [],
        'training_error_history_list': [],
        'validation_error_history_list': [],
        'best_validation_error': np.inf,
        'best_parameters': None
    }

    for result in results:
        info_dict['parameters_list'].append(result['parameters_set'])
        info_dict['training_error_history_list'].append(result['training_error_history'])
        info_dict['validation_error_history_list'].append(result['validation_error_history'])

        if result['validation_error_history'][-1] < info_dict['best_validation_error']:
            info_dict['best_validation_error'] = result['validation_error_history'][-1]
            info_dict['best_parameters'] = result['parameters_set']

    return info_dict