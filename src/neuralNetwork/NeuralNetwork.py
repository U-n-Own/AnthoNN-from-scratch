import copy
import pickle
from typing import List

import numpy as np
import os
from src.neuralNetwork.Layer import Layer
from src.neuralNetwork.error import Error, MeanSquaredError


class NeuralNetwork:
    """
    Documentazione: https://evergreen-bosworth-0ee.notion.site/Documentazione-codice-bb9e8289652d4fff9faa71d0ef93afba
    """

    layers: List[Layer] = [] # Hidden e output layer della rete (non contiene l'input layer)
    error: Error

    def __init__(self, layers: List[Layer], error: Error = MeanSquaredError()):
        if layers is None:
            raise ValueError("layers must be != None")
        if len(layers) == 0:
            raise ValueError("layers must be not empty")
        if error is None:
            raise ValueError("error must be != None")
        # Controllo che il numero d'input del layer i-esimo sia uguale al numero di neuroni del layer precedente
        for i in range(1, len(layers)):
            if layers[i].num_inputs != layers[i - 1].num_neurons:
                raise ValueError(
                    f"layers[i].num_inputs ({layers[i].num_inputs})  must be == layers[i-1].num_neurons ({layers[i - 1].num_neurons})")

        self.layers = layers
        self.error = error

    def get_weights_and_biases(self):
        """ 
        Save weights and biases of each layer in a dictionary
        :return: list of weights of each layer in dictionary form
        """
        weights = {}
        biases = {}

        for i, layer in enumerate(self.layers):
            current_layer_weights = copy.deepcopy(layer.weights)
            current_layer_biases = copy.deepcopy(layer.biases)
            weights[f"layer_{i}"] = current_layer_weights
            biases[f"layer_{i}"] = current_layer_biases


        return weights, biases

            
    def save_model(self, destination_dir: str):
        """
        Save the model in a file
        destination_dir: path of the file where the model will be saved
        """
        
        with open(destination_dir, "wb") as file:
            pickle.dump(self, file)
            
    def load_model(self, source_dir: str):
        """
        Load the model from a file
        source_dir: path of the file where the model is saved
        """
        
        with open(source_dir, "rb") as file:
            return pickle.load(file)
        

    def validate(self, validation_inputs: np.matrix, validation_outputs: np.matrix) -> np.float64:
        """ evaluate the network error on validation data """

        predicted_outputs = self.predict(validation_inputs)
        validation_error = self.error.calculate_total_error(validation_outputs, predicted_outputs)

        return validation_error

    def predict(self, inputs: np.matrix) -> np.matrix:
        """
        calcola l'output della rete neurale dato inputs

        :param inputs: matrice di inputs (dimensione: num_samples x num_features)
        :return: matrice di output (dimensione: num_samples x num_output_neurons)
        """

        if inputs is None:
            raise ValueError("inputs must be != None")
        if inputs.shape[1] != self.layers[0].num_inputs:
            raise ValueError(f"inputs.shape[1] {(inputs.shape[1])} deve "
                             f"essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def train(self, target_inputs_training: np.matrix, target_outputs_training: np.matrix,
              learning_rate: float, regularization_term: float, momentum_term: float, epochs: int,
              target_inputs_validation: np.matrix = None, target_outputs_validation: np.matrix = None) -> (List[np.float64], List[np.float64]):
        """
            Applica l'algoritmo di backpropagation su più samples alla volta

            :target_inputs_training: matrice(num_samples, num_inputs_rete_neurale) di inputs per il training
            :target_outputs_training: matrice(num_samples, num_neuroni_ultimo_layer) di outputs per il training
            :epochs: numero d'iterazioni dell'algoritmo di backpropagation

            :return: Lista degli errori sul training e, eventualmente, validation. L'i-esimo elemento contiene l'errore dell'i-esimo epoch.
        """
        self._train_parameters_checking(target_inputs_training=target_inputs_training,
                                        target_outputs_training=target_outputs_training,
                                        learning_rate=learning_rate, epochs=epochs,
                                        target_inputs_validation=target_inputs_validation,
                                        target_outputs_validation=target_outputs_validation)

        training_error_history = []
        validation_error_history = []

        for epoch in range(epochs):
            self._training_on_epoch(target_inputs_training=target_inputs_training,
                                    target_outputs_training=target_outputs_training,
                                    learning_rate=learning_rate, regularization_term=regularization_term,
                                    momentum_term=momentum_term)

            print(f"Epoch {epoch}/{epochs}: ", end = '')

            # Calcolo l'errore di training
            total_error_training = self.validate(validation_inputs=target_inputs_training,
                                                 validation_outputs=target_outputs_training)
            training_error_history.append(total_error_training)

            # Calcolo l'errore di validation
            if target_inputs_validation is not None and target_outputs_validation is not None:
                total_error_validation = self.validate(validation_inputs=target_inputs_validation,
                                                       validation_outputs=target_outputs_validation)
                validation_error_history.append(total_error_validation)

                print(f"Training error: {total_error_training} - Validation error: {total_error_validation}")
            else:
                print(f"Training error: {total_error_training}")


        return training_error_history, validation_error_history

    def _train_parameters_checking(self, target_inputs_training: np.matrix, target_outputs_training: np.matrix,
                                   learning_rate: float, epochs,
                                   target_inputs_validation: np.matrix, target_outputs_validation: np.matrix):
        """
        Funzione di supporto per la funzione train. Controlla che gli argomenti passati alla funzione train siano validi
        """

        if target_inputs_training is None and target_outputs_training is None:
            raise ValueError("target_inputs_training and target_outputs_training must be both != None")
        self._data_parameters_checking(target_inputs_training, target_outputs_training)


        if type(target_inputs_validation) != type(target_outputs_validation):
            raise ValueError("target_inputs_validation and target_outputs_validation must be both None or both != None")
        if target_inputs_validation is not None and target_outputs_validation is not None:
            self._data_parameters_checking(target_inputs_validation, target_outputs_validation)

        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 0:
            raise ValueError("epochs must be > 0")

    def _data_parameters_checking(self, target_inputs: np.matrix, target_outputs: np.matrix):
        """
        Funzione di supporto per la funzione train. Controlla che gli argomenti passati alla funzione train siano validi
        """
        
        if type(target_inputs) != np.matrix or type(target_outputs) != np.matrix:
            raise ValueError("target_inputs and target_outputs must be both np.matrix")

        if target_inputs.shape[1] != self.layers[0].num_inputs:
            raise ValueError(f"inputs.shape ({target_inputs.shape[1]}) deve essere "
                             f"uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")

        if target_outputs.shape[1] != self.layers[-1].num_neurons:
            raise ValueError( f"target_outputs.shape ({target_outputs.shape[1]}) deve essere uguale "
                              f"al numero di neuroni dell'ultimo layer ({self.layers[-1].num_neurons})")

    def _training_on_epoch(self, target_inputs_training: np.matrix, target_outputs_training: np.matrix,
                           learning_rate: float, regularization_term: float, momentum_term: float):
        """
        Funzione di supporto di "train", applica l'algoritmo di backpropagation su un singolo epoch
        """

        for layer in self.layers:
            layer.previous_delta_weight = layer.current_delta_weight.copy()
            layer.current_delta_weight = np.matrix(np.zeros(layer.current_delta_weight.shape))
            layer.current_delta_bias = np.matrix(np.zeros(layer.current_delta_bias.shape))

        # applica backpropagation su ogni sample
        for target_input, target_output in zip(target_inputs_training, target_outputs_training):
            # _backpropagation aggiorna il valore current_delta_weight
            self._backpropagation(target_input=target_input, target_output=target_output)


        learning_rate = learning_rate / target_inputs_training.shape[0] # Serve per implementare il least mean square
        # Aggiorno i pesi e i bias
        for layer in self.layers:
            # N.B.learning_rate obbligatorio anche per il momentum
            delta_weight = learning_rate * layer.current_delta_weight + momentum_term * learning_rate * layer.previous_delta_weight
            layer.weights = layer.weights + delta_weight - 2 * regularization_term * layer.weights
            layer.biases = layer.biases + learning_rate * layer.current_delta_bias

    def _backpropagation(self, target_input: np.matrix, target_output: np.matrix) -> None:
        """
            Calcola il delta_weight per ogni layer tramite l'algoritmo di backpropagation

            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_outputs: matrice(1, num_neuroni_ultimo_layer)
        """

        # Backpropagation ultimo layer
        delta_error, delta_weight = self._backpropagation_output_layer(target_input, target_output)
        self.layers[-1].current_delta_weight += delta_weight
        self.layers[-1].current_delta_bias += delta_error

        # Backpropagation hidden layer
        for i in range(len(self.layers) - 2, -1, -1):  # (scorre la lista in ordine inverso, dal penultimo al primo layer)
            delta_error, delta_weight = self._backpropagation_hidden_layer(i, target_input, delta_error)
            self.layers[i].current_delta_weight += delta_weight
            self.layers[i].current_delta_bias += delta_error

    def _backpropagation_output_layer(self, target_input: np.matrix, target_output: np.matrix) -> (np.matrix, np.matrix):
        """
            spiegazione nella sezione "Backpropagation": https://evergreen-bosworth-0ee.notion.site/Documentazione-codice-bb9e8289652d4fff9faa71d0ef93afba

            Applica algoritmo di backpropagation all'ultimo layer
            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_output: matrice(1, num_neuroni_ultimo_layer)

            :return: delta_error, delta_weight
                delta_error: matrice(1, num_neurons_ultimo layer).
                delta_weight matrice(num_neuroni_layer_precedente, num_neuroni_ultimo layer).
        """
        nn_outputs = self.predict(target_input)
        output_layer = self.layers[-1]

        derivative_vector = np.vectorize(output_layer.activation_function.derivative)(output_layer.net)
        delta_error = np.multiply((target_output-nn_outputs), derivative_vector)
        if len(self.layers) == 1:
            output_penultimate_layer = target_input
        else:
            output_penultimate_layer = self.layers[-2].output
        delta_weight = np.outer(delta_error, output_penultimate_layer)

        return delta_error, delta_weight

    def _backpropagation_hidden_layer(self, index_layer: int, target_input: np.matrix,
                                      delta_error_next_layer: np.matrix):
        """
        :index_layer: indice del layer "corrente" (i.e. layer di cui aggiorniamo i weights)
        :target_input: matrice(1, num_inputs_rete_neurale)
        :delta_error_next_layer matrice(1, num_neuroni_layer_successivo). L'i-esimo elemento è il delta_error del i-esimo neurone.
        """

        next_layer = self.layers[index_layer + 1]
        current_layer = self.layers[index_layer]

        derivative_vector = np.vectorize(current_layer.activation_function.derivative)(current_layer.net)
        delta_error = np.multiply((delta_error_next_layer * next_layer.weights), derivative_vector)

        if index_layer == 0:
            outputs_previous_layer = target_input
        else:
            outputs_previous_layer = self.layers[index_layer - 1].output
        delta_weight = np.outer(delta_error, outputs_previous_layer)

        return delta_error, delta_weight