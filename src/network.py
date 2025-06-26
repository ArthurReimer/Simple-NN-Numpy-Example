import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(activations):
    return activations * (1-activations)

class Layer:
    def __init__(self, input_amount, neuron_amount):
        self.neuron_amount = neuron_amount
        self.biases = np.empty(neuron_amount, float)
        self.weights = np.empty((neuron_amount, input_amount), float)
        self.net_inputs = np.empty(neuron_amount, float)
        self.activations = np.empty(neuron_amount, float)
        self.deltas = np.empty(neuron_amount, float)

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_amount, neuron_amount):
        new_layer = Layer(input_amount, neuron_amount)
        self.layers.append(new_layer)

    def setup(self):
        for layer in self.layers:
            layer.weights = np.random.uniform(-1, 1, layer.weights.shape)
            layer.biases = np.random.uniform(-0.5, 0.5, layer.biases.shape)

    def forward_pass(self, inputs):
        current_inputs = inputs
    

        for layer in self.layers:
            layer.net_inputs = np.dot(layer.weights, current_inputs) + layer.biases
            layer.activations = sigmoid(layer.net_inputs)
            current_inputs = layer.activations

    def backward_pass(self, inputs, learning_rate, target):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                layer.deltas = deriv_sigmoid(layer.activations) * (target - layer.activations)
            else:
                next_layer = self.layers[i + 1]
                layer.deltas = deriv_sigmoid(layer.activations) * np.dot(next_layer.weights.T, next_layer.deltas)

            prev_activations = inputs if i == 0 else self.layers[i - 1].activations

            delta_weight = learning_rate * np.outer(prev_activations, layer.deltas).T
            delta_biases = learning_rate * layer.deltas

            layer.weights += delta_weight
            layer.biases += delta_biases

    def return_output(self):
        return self.layers[-1].activations
            
def MSE_Loss(predicted, target):
    return np.mean((predicted - target) ** 2)

def flatten(mat):
    return np.array(mat).reshape(-1)

def format_target(target):
    new_arr = np.zeros(10)
    new_arr[target] = 1
    return np.array(new_arr)
