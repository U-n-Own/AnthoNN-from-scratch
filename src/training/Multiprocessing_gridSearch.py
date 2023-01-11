import multiprocessing


def grid_search(parameters_grid, target_inputs, target_outputs):
    """
    Performs a grid search on the parameters of the neural network.
    :param parameters_grid: a dictionary containing the parameters to be tested.
            Esempio di parameters_grid:
            parameters_grid = {
            'model': [neuralNetwork4],
            'learning_rate': [0.075],
            'momentum_term': [0.5],
            'regularization_term': [0],
            'epochs': [800]
            }
    :return: ritorna una lista di tuple (parametri, error_list), dove "parametri" è
            una tupla della forma: (learning_rate, momentum_term, regularization_term, epochs);
            "error_list" è la lista degli errori ottenuti durante il training
    """


    error_queue = multiprocessing.Queue()

    process_list = []

    # Loop over the different hyperparameters
    for model in parameters_grid['model']:
        for learning_rate in parameters_grid['learning_rate']:
            for momentum_term in parameters_grid['momentum_term']:
                for regularization_term in parameters_grid['regularization_term']:
                    for epochs in parameters_grid['epochs']:
                        # Definizione del task
                        args = [model, target_inputs, target_outputs, learning_rate,
                                epochs, regularization_term, momentum_term, error_queue]
                        process = multiprocessing.Process(target=_training, args=args)
                        process_list.append(process)

    for process in process_list:
        process.start()
        print("Processo {} avviato".format(process.pid))
    for process in process_list:
        process.join()
        print("Process ", process.pid, " terminated")


    models_error_list = []
    while not error_queue.empty():
        item = error_queue.get()
        models_error_list.append(item)

    return models_error_list

def _training(neuralNetwork, target_inputs, target_outputs,
             learning_rate, epochs, regularization_term,
             momentum_term, error_queue):
    error_list =  neuralNetwork.train(target_inputs=target_inputs, target_outputs=target_outputs,
                                epochs = epochs, learning_rate = learning_rate,
                                regularization_term=regularization_term, momentum_term=momentum_term)

    param = (learning_rate, momentum_term, regularization_term, epochs)
    error_queue.put((param, error_list))