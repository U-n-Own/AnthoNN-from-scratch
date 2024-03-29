import numpy as np

from src.neuralNetwork.MatrixInitialization import  ReandomInitialization
from src.neuralNetwork.function import ActivationFunction


class Layer:
    """
        Spiegazione grafica: https://evergreen-bosworth-0ee.notion.site/Documentazione-codice-bb9e8289652d4fff9faa71d0ef93afba
        N.B. l'input non viene considerato come layer
    """
    num_neurons: int = 0  # Numero di neuroni del layer
    num_inputs: int = 0  # Numero di input che riceve ogni neurone (i.e. Numero di neuroni del layer precedente)

    weights: np.matrix  # matrice dei pesi: (num_neurons x num_inputs)
    biases: np.ndarray  # vettore dei bias: (1 x num_neurons)

    activation_function: ActivationFunction # Funzione di attivazione usata da ogni neurone

    # Contiene il delta_weight restitutio dall'algoritmo di backpropagation all'epoch PRECENDETE e CORRENTE.
    # Usato per applicare il momentum (N.B. non contiene il learning rate)
    previous_delta_weight: np.matrix
    current_delta_weight: np.matrix

    current_delta_bias: np.ndarray

    net: np.matrix  # matrice (num_samples x num_neurons)
    output: np.matrix  # matrice (num_samples x num_neurons)

    def __init__(self, num_neurons: int, num_inputs: int, activation_function: ActivationFunction,
                 weightsInitialization = ReandomInitialization(-0.03, 0.2),
                 biasInitialization = ReandomInitialization(-0.01, 0.2)):

        if num_neurons <= 0:
            raise ValueError("num_neurons must be > 0")
        if num_inputs <= 0:
            raise ValueError("num_inputs must be > 0")
        if activation_function is None:
            raise ValueError("activation_function must be != None")

        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.activation_function = activation_function

        self.biases = biasInitialization.generate((1, num_neurons))
        self.weights = weightsInitialization.generate((num_neurons, num_inputs))

        self.previous_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_weight = np.matrix(np.zeros((num_neurons, num_inputs)))
        self.current_delta_bias = np.matrix(np.zeros((1, num_neurons)))

    def forward(self, inputs: np.matrix) -> np.matrix:
        """
        Dato 'inputs' calcola l'output del layer
        :param inputs: matrice di inputs (dimensione: num_samples x num_features)
        :return: matrice di output (dimensione: num_samples x num_neurons)
        """

        if inputs is None:
            raise ValueError("inputs must be != None")
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"inputs.shape[1] ({inputs.shape[1]}) must be == self.num_inputs ({self.num_inputs})")

        self.net = inputs * self.weights.T + self.biases
        self.output = np.vectorize(self.activation_function.output)(self.net)
        return self.output