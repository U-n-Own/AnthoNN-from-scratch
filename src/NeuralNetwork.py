import copy
from typing import List
import numpy as np

from src.function import ActivationFunction


class Layer:
    """
        N.B. l'input non viene considerato come layer
    """
    num_neurons: int = 0  # numero di neuroni del layer precedente
    num_inputs: int = 0  # numero di input del layer (i.e. numero di neuroni del layer precedente)

    weights: np.matrix  # matrice
    biases: np.ndarray  # vettore

    previous_delta_weight: np.matrix = None # Contiene il delta_weight restitutio dall'algoritmo di backpropagation all'epoch precedente. Usato per applicare il momentum (N.B. non contiene il learning rate)
    current_delta_weight: np.matrix = None # Contiene il delta_weight restitutio dall'algoritmo di backpropagation all'epoch corrente. Usato per applicare il momentum (N.B. non contiene il learning rate)

    current_delta_bias: np.ndarray = None

    net: np.ndarray = None  # matrice (num_neurons x 1)
    output: np.ndarray = None  # matrice (num_neurons x 1)

    activation_function: ActivationFunction

    def __init__(self, num_neurons: int, num_inputs: int,
                 activation_function: ActivationFunction,
                 min_value_weight=0.01, max_value_weight=0.2,
                 min_value_bias=0.01, max_value_bias=0.2):
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
        self.weights = np.matrix(np.random.uniform(low=min_value_weight, high=max_value_weight, size=(num_neurons, num_inputs)))


        self.previous_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_bias = np.matrix(np.zeros((1, num_neurons)))


        # set random biases
        # TODO Per ora lasciare biases = 0. Dopo controllare come viene modificato l'algoritmo
        # di backpropagation con biases != 0 - (bisogna derivare anche per i ogni b \in biases)
        self.biases = np.matrix(np.random.uniform(low=min_value_bias, high=max_value_bias, size=num_neurons))

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

    def __init__(self, layers: [Layer]):
        if layers is None:
            raise ValueError("layers must be != None")
        if len(layers) == 0:
            raise ValueError("layers must be not empty")

        # Controllo che il numero d'input del layer i-esimo sia uguale al numero di neuroni del layer precedente
        for i in range(1, len(layers)):
            if layers[i].num_inputs != layers[i - 1].num_neurons:
                raise ValueError(f"layers[i].num_inputs ({layers[i].num_inputs})  must be == layers[i-1].num_neurons ({layers[i-1].num_neurons})")

        self.layers = layers

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

    def train(self, target_inputs: np.matrix, target_outputs: np.matrix, learning_rate: float, regularization_term: float, momentum_term: float, epochs: int) -> List[np.float64]:
        """
            Applica l'algoritmo di backpropagation su più samples alla volta

            :target_input: matrice(num_samples, num_inputs_rete_neurale)
            :target_outputs: matrice(num_samples, num_neuroni_ultimo_layer)
            :epochs: numero di iterazioni dell'algoritmo di backpropagation

            :return: lista di errori. L'i-esimo elemento contiene l'errore dell'i-esimo epoch. L'errore viene calcolato con calculate_total_error
        """
        if target_inputs is None:
            raise ValueError("inputs must be != None")
        if target_inputs.shape[1] != self.layers[0].num_inputs:
            raise ValueError(f"inputs.shape ({target_inputs.shape[1]}) deve essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")

        if target_outputs is None:
            raise ValueError("target_outputs must be != None")
        if target_outputs.shape[1] != self.layers[-1].num_neurons:
            raise ValueError(f"target_outputs.shape ({target_outputs.shape[1]}) deve essere uguale al numero di neuroni dell'ultimo layer ({self.layers[-1].num_neurons})")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 0:
            raise ValueError("epochs must be >= 0")


        error_history = []

        try:
            for _ in range(epochs):
                for layer in self.layers:
                    layer.previous_delta_weight = layer.current_delta_weight.copy()
                    layer.current_delta_weight = np.matrix(np.zeros(layer.current_delta_weight.shape))

                    layer.current_delta_bias = np.matrix(np.zeros(layer.current_delta_bias.shape))

                for target_input, target_output in zip(target_inputs, target_outputs):
                    #_backpropagation aggiorna il valore current_delta_weight
                    self._backpropagation(target_input=target_input, target_output=target_output)

                for layer in self.layers:
                    layer.weights = layer.weights + learning_rate*layer.current_delta_weight + \
                                    learning_rate*momentum_term*layer.previous_delta_weight - \
                                    2*regularization_term*layer.weights
                    layer.biases = layer.biases + learning_rate*layer.current_delta_bias

                output_nn = self.predict(inputs=target_inputs)
                error_history.append(calculate_total_error(target_output=target_outputs, output_nn=output_nn))

        except OverflowError:
            raise OverflowError("Errore di overflow durante il training")

        return error_history


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
        for i in range(len(self.layers) - 2, -1, -1):  # (scorre la lista in ordine inverso, dal penultimo al primo layer)
            delta_error, delta_weight = self._backpropagation_hidden_layer(i, target_input, delta_error)
            self.layers[i].current_delta_weight += delta_weight
            self.layers[i].current_delta_bias += delta_error

    @staticmethod
    def _check_overflow(weight: np.matrix) -> None:
        """
        Funzione di supporto per "_backpropagation": controlla se ci sono overflow
        """
        if np.isinf(weight).any():
            raise OverflowError("Overflow durante il backpropagation")

    def _backpropagation_output_layer(self, target_input: np.matrix, target_output: np.matrix) -> (np.matrix, np.matrix):
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
        delta_error = np.multiply((target_output - outputs), derivative_vector) # np.multiply: element-wise product
        delta_weight = np.outer(delta_error, output_penultimate_layer)

        return delta_error, delta_weight

    def _backpropagation_hidden_layer(self, index_layer: int, target_input: np.matrix, delta_error_next_layer: np.matrix):
        """
        Spiegazione: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#d86bff7059f0498397a8cdeae1c1631d
        :index_layer: indice del layer "corrente" (i.e. layer di cui aggiorniamo i weights)
        :target_input: matrice(1, num_inputs_rete_neurale)
        :delta_error_next_layer matrice(1, num_neuroni_layer_successivo). L'i-esimo elemento è il delta_error del i-esimo neurone.
        """

        if target_input is None:
            raise ValueError("inputs must be != None")
        if target_input.shape != (1, self.layers[0].num_inputs):
            raise ValueError(f"inputs.shape ({target_input.shape}) deve essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")
        if delta_error_next_layer is None:
            raise ValueError("delta_error_next_layer must be != None")
        if delta_error_next_layer.shape != (1, self.layers[index_layer+1].num_neurons):
            raise ValueError(f"delta_error_next_layer.shape ({delta_error_next_layer.shape}) deve essere uguale al numero di neuroni del layer successivo ({self.layers[-1].num_neurons})")

        next_layer = self.layers[index_layer + 1]
        current_layer = self.layers[index_layer]
        if index_layer == 0:
            outputs_previous_layer = target_input
        else:
            outputs_previous_layer = self.layers[index_layer - 1].output

        derivative_vector = np.vectorize(current_layer.activation_function.derivative)(current_layer.net)
        delta_error = np.multiply((delta_error_next_layer * next_layer.weights), derivative_vector)  # np.multiply: element-wise product
        delta_weight = np.outer(delta_error, outputs_previous_layer)

        return delta_error, delta_weight


def calculate_total_error(target_output: np.matrix, output_nn: np.matrix) -> np.float64:
    """
    implementazione dell'errore spiegato a lezione: https://www.notion.so/Back-propagation-c35b14a0570246ebaf5f82722a0cb8a3#3bbf8fb43aa94a0a98d29d80076d79e3

    :param target_output: matrice dell'output desiderato (dimensione: num_samples x num_output_neurons)
    :param output_nn: matrice dell'output della rete neurale (dimensione: num_samples x num_output_neurons)

    :return: errore totale
    """
    if target_output is None:
        raise ValueError("target_output must be != None")
    if output_nn is None:
        raise ValueError("output_nn must be != None")
    if target_output.shape != output_nn.shape:
        raise ValueError(f"target_output ({target_output.shape}) and output_nn ({output_nn.shape}) must have the same shape")

    error_vector = np.sum(np.square(target_output - output_nn), axis=1)
    error_total = np.sum(error_vector)
    return error_total