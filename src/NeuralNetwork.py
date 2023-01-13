from typing import List
import numpy as np

from src.neuralNetwork.error import Error, MeanSquaredError
from src.neuralNetwork.function import ActivationFunction


class Layer:
    """
        N.B. l'input non viene considerato come layer
    """
    num_neurons: int = 0  # numero di neuroni del layer precedente
    num_inputs: int = 0  # numero di input del layer (i.e. numero di neuroni del layer precedente)

    weights: np.matrix  # matrice
    biases: np.ndarray  # vettore

    previous_delta_weight: np.matrix = None  # Contiene il delta_weight restitutio dall'algoritmo di backpropagation all'epoch precedente. Usato per applicare il momentum (N.B. non contiene il learning rate)
    current_delta_weight: np.matrix = None  # Contiene il delta_weight restitutio dall'algoritmo di backpropagation all'epoch corrente. Usato per applicare il momentum (N.B. non contiene il learning rate)

    current_delta_bias: np.ndarray = None

    net: np.ndarray = None  # matrice (num_neurons x 1)
    output: np.ndarray = None  # matrice (num_neurons x 1)

    activation_function: ActivationFunction

    def __init__(self, num_neurons: int, num_inputs: int,
                 activation_function: ActivationFunction,
                 min_value_weight=-0.03, max_value_weight=0.2,
                 min_value_bias=-0.01, max_value_bias=0.2):
        """
        :param num_neurons: numero di neuroni del layer
        :param num_inputs: numero input di ogni unit (i.e. Numero di neuroni del layer precedente)
        """

        if num_neurons <= 0:
            raise ValueError("num_neurons must be > 0")
        if num_inputs <= 0:
            raise ValueError("num_inputs must be > 0")
        if activation_function is None:
            raise ValueError("activation_function must be != None")
        if min_value_weight > max_value_weight:
            raise ValueError(f"min_value_weight must be ({min_value_weight}) <= max_value_weight ({min_value_weight})")
        if min_value_bias > max_value_bias:
            raise ValueError(f"min_value_bias must be ({min_value_bias}) <= max_value_bias ({max_value_bias}) ")

        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.activation_function = activation_function

        # set random biases and weights
        self.biases = np.matrix(np.random.uniform(low=min_value_bias, high=max_value_bias, size=(1, num_neurons)))
        self.weights = np.matrix(
            np.random.uniform(low=min_value_weight, high=max_value_weight, size=(num_neurons, num_inputs)))

        self.previous_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_bias = np.matrix(np.zeros((1, num_neurons)))

    def calculate_outputs(self, inputs: np.matrix) -> np.matrix:
        """
        calcola l'output del layer dato inputs
        :param inputs: matrice di inputs (dimensione: num_samples x num_features)
        :return: matrice di output (dimensione: num_samples x num_output_neurons)
        (spiegazione grafica: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#b7fb73f5f1ac43deb096fd64f968c540)

        """

        if inputs is None:
            raise ValueError("inputs must be != None")
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"inputs.shape[1] ({inputs.shape[1]}) must be == self.num_inputs ({self.num_inputs})")

        self.net = inputs * self.weights.T + self.biases
        self.output = np.vectorize(self.activation_function.output)(self.net)
        return self.output


class NeuralNetwork:
    layers: List[Layer] = []
    error: Error

    def __init__(self, layers: [Layer], error: Error = MeanSquaredError()):
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

    def get_weights(self):
        # TODO controllare se fa una shallow copy
        # TODO save weights in a file for later use
        """ Method used to save final weights of our model

        :return: list of weights of each layer in dictionary form
        """
        weights = {}
        current_layer_weights = []

        for layer in self.layers:
            current_layer_weights.append(layer.weights)
            weights.update({layer: current_layer_weights})

        return weights

    def set_weights(self, weights: dict):
        """ Method used to set/load weights of our trained model """
        
        # Check if number of layer of current net is equal to the number of layer we have in the dictionary
        if len(self.layers) != len(weights):
            raise ValueError(f"Number of layers in the network ({len(self.layers)}) must be equal to the number of layers of loaded weights ({len(weights)})")
        
        for layer in self.layers:
            layer.weights = weights[layer]
        
    
    def validate(self, validation_inputs: np.matrix, validation_outputs: np.matrix) -> float:
        # TODO controllo argomenti
        """ Evaluating the network error on validation data """

        predicted_outputs = self.predict(validation_inputs)
        validation_error = self.error.calculate_total_error(validation_outputs, predicted_outputs)

        return validation_error

    def predict(self, inputs: np.matrix) -> np.matrix:
        """
        calcola l'output della rete neurale dato inputs

        :param inputs: matrice di inputs (dimensione: num_samples x num_features)
        :return: matrice di output (dimensione: num_samples x num_output_neurons)
        (spiegazione grafica: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#b7fb73f5f1ac43deb096fd64f968c540)
        """

        if inputs is None:
            raise ValueError("inputs must be != None")
        if inputs.shape[1] != self.layers[0].num_inputs:
            raise ValueError(f"inputs.shape[1] {(inputs.shape[1])} deve "
                             f"essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")

        for layer in self.layers:
            outputs = layer.calculate_outputs(inputs)
            inputs = outputs

        return inputs

    def train(self, target_inputs_training: np.matrix, target_outputs_training: np.matrix,
              learning_rate: float, regularization_term: float, momentum_term: float, epochs: int,
              target_inputs_validation: np.matrix = None, target_outputs_validation: np.matrix = None) -> (List[np.float64], List[np.float64]):
        """
            Applica l'algoritmo di backpropagation su più samples alla volta

            :target_inputs_training: matrice(num_samples, num_inputs_rete_neurale) di inputs per il training
            :target_outputs_training: matrice(num_samples, num_neuroni_ultimo_layer) di outputs per il training
            :epochs: numero di iterazioni dell'algoritmo di backpropagation

            :return: lista di errori. L'i-esimo elemento contiene l'errore dell'i-esimo epoch. L'errore viene calcolato con mean_squared_error
        """
        self._train_arguments_checking(target_inputs_training=target_inputs_training,
                                       target_outputs_training=target_outputs_training,
                                       learning_rate=learning_rate, epochs=epochs,
                                       target_inputs_validation=target_inputs_validation,
                                       target_outputs_validation=target_outputs_validation)

        training_error_history = []
        validation_error_history = []

        for _ in range(epochs):
            self._training_on_epoch(target_inputs_training=target_inputs_training,
                                    target_outputs_training=target_outputs_training,
                                    learning_rate=learning_rate, regularization_term=regularization_term,
                                    momentum_term=momentum_term)

            # Calcolo l'errore di training
            total_error_training = self.validate(validation_inputs=target_inputs_training,
                                                 validation_outputs=target_outputs_training)
            training_error_history.append(total_error_training)

            # Calcolo l'errore di validation
            if target_inputs_validation is not None and target_outputs_validation is not None:
                total_error_validation = self.validate(validation_inputs=target_inputs_validation,
                                                       validation_outputs=target_outputs_validation)
                validation_error_history.append(total_error_validation)


        return training_error_history, validation_error_history

    def _train_arguments_checking(self, target_inputs_training: np.matrix, target_outputs_training: np.matrix,
                                  learning_rate: float, epochs,
                                  target_inputs_validation: np.matrix, target_outputs_validation: np.matrix):
        """
        Funzione di supporto per la funzione train. Controlla che gli argomenti passati alla funzione train siano validi
        """


        if target_inputs_training is None:
            raise ValueError("inputs must be != None")
        if target_inputs_training.shape[1] != self.layers[0].num_inputs:
            raise ValueError(
                f"inputs.shape ({target_inputs_training.shape[1]}) deve essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")

        if target_outputs_training is None:
            raise ValueError("target_outputs must be != None")
        if target_outputs_training.shape[1] != self.layers[-1].num_neurons:
            raise ValueError(
                f"target_outputs.shape ({target_outputs_training.shape[1]}) deve essere uguale al numero di neuroni dell'ultimo layer ({self.layers[-1].num_neurons})")

        if type(target_inputs_validation) != type(target_outputs_validation):
            raise ValueError("target_inputs_validation and target_outputs_validation must be both None or both != None")
        if target_inputs_validation is not None and target_outputs_validation is not None:
            if target_inputs_training.shape[1] != target_inputs_validation.shape[1]:
                raise ValueError(f"target_inputs_training.shape[1] ({target_inputs_training.shape[1]}) deve essere "
                                 f"uguale a target_inputs_validation.shape[1] ({target_inputs_validation.shape[1]})")
            if target_outputs_training.shape[1] != target_outputs_validation.shape[1]:
                raise ValueError(f"target_outputs_training.shape[1] ({target_outputs_training.shape[1]}) deve essere "
                                 f"uguale a target_outputs_validation.shape[1] ({target_outputs_validation.shape[1]})")
            if target_inputs_validation.shape[0] != target_outputs_validation.shape[0]:
                raise ValueError(f"target_inputs_validation.shape[0] ({target_inputs_validation.shape[0]}) deve essere "
                                 f"uguale a target_outputs_validation.shape[0] ({target_outputs_validation.shape[0]})")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 0:
            raise ValueError("epochs must be >= 0")

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

        # Aggiorno i pesi e i bias
        for layer in self.layers:
            layer.weights = layer.weights + learning_rate * layer.current_delta_weight + \
                            learning_rate * momentum_term * layer.previous_delta_weight - \
                            2 * regularization_term * layer.weights
            layer.biases = layer.biases + learning_rate * layer.current_delta_bias

    def _backpropagation(self, target_input: np.matrix, target_output: np.matrix) -> None:
        """
            Calcola il delta_weight tramite l'algoritmo di backpropagation

            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_outputs: matrice(1, num_neuroni_ultimo_layer)
        """

        # Backpropagation ultimo layer
        delta_error, delta_weight = self._backpropagation_output_layer(target_input, target_output)
        self.layers[-1].current_delta_weight += delta_weight
        self.layers[-1].current_delta_bias += delta_error

        # Backpropagation hidden layer
        for i in range(len(self.layers) - 2, -1,
                       -1):  # (scorre la lista in ordine inverso, dal penultimo al primo layer)
            delta_error, delta_weight = self._backpropagation_hidden_layer(i, target_input, delta_error)
            self.layers[i].current_delta_weight += delta_weight
            self.layers[i].current_delta_bias += delta_error

    def _backpropagation_output_layer(self, target_input: np.matrix, target_output: np.matrix) -> (
    np.matrix, np.matrix):
        """
            spiegazione: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#bd5c48fceacb464a81acda5ce37a2433

            Applica algoritmo di backpropagation all'ultimo layer
            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_output: matrice(1, num_neuroni_ultimo_layer)

            :return: delta_error, delta_weight
            :delta_error matrice(1, num_neurons_ultimo layer). L'i-esimo elemento è il delta_error del i-esimo neurone.
            :delta_weight matrice(num_neuroni_layer_precedente, num_neuroni_ultimo layer).
                          L'i-esimo elemento è il delta_weight del i-esimo neurone
                          (come indicato qui: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#91e92e1fbf8648b8880aadeb95c6770c)
        """

        outputs = self.predict(target_input)
        output_layer = self.layers[-1]

        if len(self.layers) == 1:
            output_penultimate_layer = target_input
        else:
            output_penultimate_layer = self.layers[-2].output

        derivative_vector = np.vectorize(output_layer.activation_function.derivative)(output_layer.net)
        delta_error = np.multiply((target_output - outputs), derivative_vector)  # np.multiply: element-wise product
        delta_weight = np.outer(delta_error, output_penultimate_layer)

        return delta_error, delta_weight

    def _backpropagation_hidden_layer(self, index_layer: int, target_input: np.matrix,
                                      delta_error_next_layer: np.matrix):
        """
        Spiegazione: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#d86bff7059f0498397a8cdeae1c1631d
        :index_layer: indice del layer "corrente" (i.e. layer di cui aggiorniamo i weights)
        :target_input: matrice(1, num_inputs_rete_neurale)
        :delta_error_next_layer matrice(1, num_neuroni_layer_successivo). L'i-esimo elemento è il delta_error del i-esimo neurone.
        """

        if target_input is None:
            raise ValueError("inputs must be != None")
        if target_input.shape != (1, self.layers[0].num_inputs):
            raise ValueError(
                f"inputs.shape ({target_input.shape}) deve essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")
        if delta_error_next_layer is None:
            raise ValueError("delta_error_next_layer must be != None")
        if delta_error_next_layer.shape != (1, self.layers[index_layer + 1].num_neurons):
            raise ValueError(
                f"delta_error_next_layer.shape ({delta_error_next_layer.shape}) deve essere uguale al numero di neuroni del layer successivo ({self.layers[-1].num_neurons})")

        next_layer = self.layers[index_layer + 1]
        current_layer = self.layers[index_layer]
        if index_layer == 0:
            outputs_previous_layer = target_input
        else:
            outputs_previous_layer = self.layers[index_layer - 1].output

        derivative_vector = np.vectorize(current_layer.activation_function.derivative)(current_layer.net)
        delta_error = np.multiply((delta_error_next_layer * next_layer.weights),
                                  derivative_vector)  # np.multiply: element-wise product
        delta_weight = np.outer(delta_error, outputs_previous_layer)

        return delta_error, delta_weight