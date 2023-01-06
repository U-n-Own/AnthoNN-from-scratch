#Code for test the function
import unittest
import numpy as np


from NeuralNetwork import NeuralNetwork, Layer, calculate_total_error
from function import SigmoideFunction, IdentityFunction


class TestNeuralNetwork(unittest.TestCase):
    def test_init(self):
        # Test initializing a neural network with 2 input units, 3 hidden units, and 1 output unit
        
        layer_1 = Layer(num_neurons = 3, num_inputs = 2, activation_function = SigmoideFunction())
        layer_2 = Layer(num_neurons = 3, num_inputs = 3, activation_function = SigmoideFunction())
        layer_3 = Layer(num_neurons = 1, num_inputs = 3, activation_function = SigmoideFunction())
        
        nn = NeuralNetwork([layer_1, layer_2, layer_3])
        
        self.assertEqual(layer_1.num_inputs, 2)
        self.assertEqual(layer_1.num_neurons, 3)
        
        self.assertEqual(layer_2.num_inputs, 3)
        self.assertEqual(layer_2.num_neurons, 3)
        
        self.assertEqual(layer_3.num_inputs, 3)
        self.assertEqual(layer_3.num_neurons, 1)
        
        
        self.assertEqual(len(nn.layers), 3)
        
        # Test that the weights and biases are randomly initialized within the specified range      
        
        self.assertGreaterEqual(np.min(nn.layers[0].weights), 0.01)
        self.assertLessEqual(np.max(nn.layers[0].weights), 0.1)
        self.assertGreaterEqual(np.min(nn.layers[0].biases), 0)
        self.assertLessEqual(np.max(nn.layers[0].biases), 0)
        
        self.assertGreaterEqual(np.min(nn.layers[1].weights), 0.01)
        self.assertLessEqual(np.max(nn.layers[1].weights), 0.1)
        self.assertGreaterEqual(np.min(nn.layers[1].biases), 0)
        self.assertLessEqual(np.max(nn.layers[1].biases), 0)
        
        self.assertGreaterEqual(np.min(nn.layers[2].weights), 0.01)
        self.assertLessEqual(np.max(nn.layers[2].weights), 0.1)
        self.assertGreaterEqual(np.min(nn.layers[2].biases), 0)
        self.assertLessEqual(np.max(nn.layers[2].biases), 0)
        
