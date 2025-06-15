import random
import numpy as np
# import time
# import tensorflow as tf

# a = np.empty((3,2), float)
# a.fill(0)
# print(a)
# print(a.shape)
# b = np.random.uniform(0.0, 1.0, list(a.shape))
# print(b)

# c = np.empty(5, float)
# c.fill(0)
# print(c)
# print(c.shape)

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

class NN:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_amount, neuron_amount):
        new_layer = Layer(input_amount, neuron_amount)
        self.layers.append(new_layer)

    def setup(self):
        for layer in self.layers:
            layer.weights = np.random.uniform(-0.5, 0.5, layer.weights.shape)
            layer.biases = np.random.uniform(-0.5, 0.5, layer.biases.shape)

    def forward_pass(self, inputs):
        current_inputs = inputs

        for layer in self.layers:
            layer.net_inputs = np.dot(layer.weights, current_inputs) + layer.biases
            current_inputs = layer.net_inputs
            layer.activations = sigmoid(layer.net_inputs)

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
            delta_biases = learning_rate * layer.biases

            layer.weights += delta_weight
            layer.biases += delta_biases
            
def binary_cross_entropy(y_pred, y_true, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


inputs = np.array([0.1, 0.2, 0.1, 0.3, 0.5])


nn = NN()
nn.add_layer(5, 4) # Hidden layer
nn.add_layer(4, 2) # Output layer
nn.setup()
nn.forward_pass(inputs)
nn.backward_pass(inputs, 0.01, [0.5, 0.2])







# class NeuronLayer:
#     def __init__(self, neuron_amount):
#         self.neurons = []
#         self.weights = np.array([])
#         self.neuron_amount = neuron_amount
#         self.neuron_activations = np.array([])
#         self.net_inputs = np.array([])
#         self.bias = np.array([])
#         self.delta_values = np.array([])
#         self.derivatives = np.array([])

# class NeuralNetwork:
#     def __init__(self, layers, input_length):
#         self.layers = layers
#         self.input_length = input_length
#         self.inputs = None
#         self.current_inputs = None

#     def create_random_weights(self):
#         for l_index, layer in enumerate(self.layers):
#             layer_weights = []

#             if l_index == 0:
#                 for _ in range(0, layer.neuron_amount):
#                     neuron_weights = []
#                     for _ in range(0, self.input_length):
#                         neuron_weights.append((random.random()-0.5)*2)
#                     layer_weights.append(neuron_weights)
#             else:
#                 for _ in range(0, layer.neuron_amount):
#                     neuron_weights = []
#                     for _ in range(0, self.layers[l_index - 1].neuron_amount):
#                         neuron_weights.append((random.random()-0.5)*2)
#                     layer_weights.append(neuron_weights)

#             layer.weights = layer_weights

#     def create_random_biases(self):
#         for _, layer in enumerate(self.layers):
#             layer_biases = []
#             for _ in range(layer.neuron_amount):
#                 layer_biases.append((random.random()-0.5)*2)
#             layer.biases = np.array(layer_biases)

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
    
#     def sigmoid_deriv(self, x):
#         return self.sigmoid(x) * (1 - self.sigmoid(x))

#     def forward_pass(self, inputs: np.array):
#         self.current_inputs = inputs
#         self.inputs = inputs

#         for layer_index, layer in enumerate(self.layers):
#             net_inputs = []
#             neuron_activations = []

#             if layer_index == 0:
#                 for neuron_index in range(0, layer.neuron_amount):
#                     net_input = sum(layer.weights[neuron_index] * self.current_inputs) + layer.biases[neuron_index]
#                     net_inputs.append(net_input)

#                     activation = self.sigmoid(net_input)
#                     neuron_activations.append(activation)
#             else:
#                 previous_layer = self.layers[layer_index-1]

#                 for neuron_index in range(0, layer.neuron_amount):
#                     net_input = sum(layer.weights[neuron_index] * previous_layer.activations) + layer.biases[neuron_index]
#                     net_inputs.append(net_input)

#                     activation = self.sigmoid(net_input)
#                     neuron_activations.append(activation)
            
#             layer.net_inputs = np.array(net_inputs)
#             layer.activations = np.array(neuron_activations)
#             self.current_inputs = layer.activations

#     def return_output(self):
#         return self.layers[-1].activations

#     def backwards_pass(self, learning_rate: float, expected_values: np.array):
#         reversed_layers = list(reversed(self.layers))

#         for layer_index, layer in enumerate(reversed_layers):
#             layer.derivatives = self.sigmoid_deriv(layer.net_inputs)

#             # delta Calculation
#             # Quelle: https://www.youtube.com/watch?v=EAtQCut6Qno&t=266s

#             # Output layer: δ = f'(netzinput) * (a(soll) - a(ist))
#             if layer_index == 0:
#                 layer.delta_values = (expected_values - layer.activations) * layer.derivatives
#             # Hidden layer: δ = f'(netzinput) * Σ(δ * w)
#             else:
#                 delta_values = []

#                 previous_layer = reversed_layers[layer_index - 1]

#                 for neuron_index in range(layer.neuron_amount):
#                     weighted_sum = 0

#                     for neuron in range(previous_layer.neuron_amount):
#                         weighted_sum += previous_layer.weights[neuron][neuron_index] * previous_layer.delta_values[neuron]
                    
#                     delta = weighted_sum * layer.derivatives[neuron_index]
#                     delta_values.append(delta)

#                 layer.delta_values = np.array(delta_values)


#             # delta weight Calculation + weight adjustment
#             if layer_index == len(self.layers)-1:

#                 for neuron_index in range(layer.neuron_amount):

#                     for weight_index in range(len(layer.weights[neuron_index])):
#                         delta_weight = learning_rate * layer.delta_values[neuron_index] * self.sigmoid(self.inputs[weight_index])
#                         layer.weights[neuron_index][weight_index] += delta_weight
                        
#             else:
#                 next_layer = reversed_layers[layer_index+1]

#                 for neuron_index in range(layer.neuron_amount):

#                     for weight_index in range(len(layer.weights[neuron_index])):
#                         delta_weight = learning_rate * layer.delta_values[neuron_index] * next_layer.activations[neuron_index]
#                         layer.weights[neuron_index][weight_index] += delta_weight

#             for neuron_index in range(layer.neuron_amount):
#                 delta_bias = learning_rate * layer.delta_values[neuron_index]
#                 layer.biases[neuron_index] += delta_bias
            
#             self.current_inputs = layer.activations

# def MSE_Loss(predicted, target):
#     return sum((predicted - target)*(predicted - target))

# def flatten(matrix):
#     new_array = []

#     for array in matrix:
#         for number in array:
#             new_array.append(number)

#     return np.array(new_array)

# def format_target(target):
#     new_array = np.zeros(10)
#     new_array[target] = 1
#     return np.array(new_array)




# mnist = tf.keras.datasets.mnist

# (train_X, train_y), (test_X, test_y) = mnist.load_data()

# train_X = train_X / 255.0
# test_X = test_X / 255.0

# inputs = np.array(flatten(train_X[0]))

# NN = NeuralNetwork(
#     [NeuronLayer(784),NeuronLayer(400),NeuronLayer(10)],
#     len(inputs)
# )

# NN.create_random_weights()
# NN.create_random_biases()


# correct_predictions = 0
# total_samples = len(train_X)

# for i in range(0, len(train_X)):
#     NN.forward_pass(flatten(train_X[i]))
#     output = NN.return_output()
#     prediction = np.argmax(output)

#     if prediction == train_y[i]:
#         correct_predictions += 1

#     NN.backwards_pass(0.01, format_target(train_y[i]))

#     loss = MSE_Loss(predicted=output, target=format_target(train_y[i])) / 10
#     print(f"Loss: {loss}, Prediction: {prediction}, Actual: {train_y[i]}")

# accuracy = correct_predictions / total_samples
# print(f"\nFinal Accuracy after {total_samples} samples: {accuracy * 100:.2f}%")
