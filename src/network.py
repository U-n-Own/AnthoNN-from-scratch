"""
We divide our neural network into three classes:
    1. Neural Network class 
    2. Layer class
    3. Neuron class
"""

# Imports standard libraries
import numpy as np

# Import our own modules
#TODO

class Network:

    def __init__(self, layers):
        self.layers = layers
    """
    This module contains the bare bone implementation of a network as we seen at lesson
    a neural networks is a compound of units divided by layers: Input, Hidden and Ouput each unit
    """

class Layer:

    def __init__(self, units):
        self.units = units
    """
    A Layer is a collection of units, we have to distinguishin between input, hidden and output layers
    """

class Perceptron:
    
    def __init__(self, weights, bias, activation_fuction):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_fuction
    """
    A perceptron is a single unit, it has a set of weights and a bias and an activation function
    """


