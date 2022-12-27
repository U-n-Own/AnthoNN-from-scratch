from typing import List
import numpy as np
from NeuralNetwork.function import IdentityFunction, SigmoideFunction


# TODO ci potrebbero essere degli errori con i tipi di numpy (e.g. np.float64, np.ndarray, np.matrix) -> fare testing
# TODO testare maggiormente neural network (soprattuto per i casi estremi)

class Layer:
    """
        N.B. l'input non viene considerato come layer
    """
    num_neurons: int = 0  # numero di neuroni del layer precedente
    num_inputs: int = 0  # numero di input del layer (i.e. numero di neuroni del layer precedente)

    weights: np.matrix  # matrice
    biases: np.ndarray  # vettore

    net: np.ndarray = None  # matrice (num_neurons x 1)
    output: np.ndarray = None  # matrice (num_neurons x 1)

    activation_function = IdentityFunction # TODO creare superclasse activation_function e creare due sottoclassi: SigmoideFunction e IdentityFunction

    def __init__(self, num_neurons: int, num_inputs: int,
                 min_value_weight=0.01, max_value_weight=0.1,
                 min_value_bias=0, max_value_bias=0):
        """
        :param num_neurons: numero di neuroni del layer
        :param num_inputs: numero input di ogni unit (i.e. numero di neuroni del layer precedente)
        """
        if num_neurons <= 0:
            raise ValueError("num_neurons must be > 0")
        if num_inputs <= 0:
            raise ValueError("num_inputs must be > 0")

        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        # set random weights
        self.weights = np.matrix(np.random.uniform(low=min_value_weight, high=max_value_weight, size=(num_neurons, num_inputs)))

        # set random biases
        #TODO cambiare il valore dei biases iniziale e vedere come cambiarli con l'algoritmo di backpropagation
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

    # TODO per ora applica l'algoritmo di backpropagation una volta sola e solo per un input
    def train(self, target_input: np.matrix, target_output: np.matrix, learning_rate: float) -> None:
        """
            Cambia il valore dei weights attraverso l'algoritmo di backpropagation

            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_outputs: matrice(1, num_neuroni_ultimo_layer)
        """

        if target_input is None:
            raise ValueError("inputs must be != None")
        if target_input.shape != (1, self.layers[0].num_inputs):
            raise ValueError(f"inputs.shape ({target_input.shape}) deve essere uguale al numero di input accettati dalla rete neurale ({self.layers[0].num_inputs})")
        if target_output is None:
            raise ValueError("target_outputs must be != None")
        if target_output.shape != (1, self.layers[-1].num_neurons, ):
            raise ValueError(f"target_outputs.shape ({target_output.shape}) deve essere uguale al numero di neuroni dell'ultimo layer ({self.layers[-1].num_neurons})")

        # Backpropagation ultimo layer
        delta_error, delta_weight = self._backpropagation_output_layer(target_input, target_output)
        self.layers[-1].weights += learning_rate * delta_weight

        # Backpropagation hidden layer
        for i in range(len(self.layers) - 2, -1, -1):  # (scorre la lista in ordine inverso, dal penultimo al primo layer)
            delta_error, delta_weight = self._backpropagation_hidden_layer(i, target_input, delta_error)
            self.layers[i].weights += learning_rate * delta_weight

    def _backpropagation_output_layer(self, target_input: np.matrix, target_output: np.matrix) -> (np.matrix, np.matrix):
        """
            spiegazione: https://www.notion.so/Documentazione-codice-837f9c910a8447658ec6a565e3259d52#bd5c48fceacb464a81acda5ce37a2433

            Applica algoritmo di backpropagation all'ultimo layer
            :target_input: matrice(1, num_inputs_rete_neurale)
            :target_output: matrice(1, num_neuroni_ultimo_layer)

            :return: delta_error, delta_weight
            :delta_error matrice(1, num_neurons_ultimo layer). l'i-esimo elemento è il delta_error del i-esimo neurone.
            :delta_weight matrice(num_neuroni_layer_precedente, num_neuroni_ultimo layer).
                          l'i-esimo elemento è il delta_weight del i-esimo neurone
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
        :delta_error_next_layer matrice(1, num_neuroni_layer_successivo). l'i-esimo elemento è il delta_error del i-esimo neurone.
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


# TODO forse conviene gestire l'input come vettore e chiamare più volte questo metodo (NO)
# TODO testare se funziona dopo le modifiche
def calculate_total_error(target_output: np.ndarray, output_nn: np.ndarray) -> np.float64:
    """
    implementazione dell'errore spiegato a lezione: https://www.notion.so/Back-propagation-c35b14a0570246ebaf5f82722a0cb8a3#3bbf8fb43aa94a0a98d29d80076d79e3

    :param target_output: output desiderato
    :param output_nn: output della rete neurale
    :return: errore totale
    """
    if target_output is None:
        raise ValueError("target_output must be != None")
    if output_nn is None:
        raise ValueError("output_nn must be != None")
    if target_output.shape != output_nn.shape:
        raise ValueError("target_output and output_nn must have the same shape")

    error_vector = np.sum(np.square(target_output - output_nn), axis=1) * 0.5  # il p-esimo elemento rappresenta E_p (https://www.notion.so/Back-propagation-c35b14a0570246ebaf5f82722a0cb8a3#3bbf8fb43aa94a0a98d29d80076d79e3)
    error_total = np.sum(error_vector)
    return error_total
