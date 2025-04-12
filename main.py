import random
import numpy as np
import time
import tensorflow as tf

class NeuronLayer:
    def __init__(self, neuron_amount):
        self.neurons = []
        self.weights = np.array([])
        self.neuron_amount = neuron_amount
        self.neuron_activations = np.array([])
        self.net_inputs = np.array([])
        self.bias = np.array([])
        self.delta_values = np.array([])
        self.derivatives = np.array([])

class NeuralNetwork:
    def __init__(self, layers, input_length):
        self.layers = layers
        self.input_length = input_length
        self.current_inputs = None

    def create_random_weights(self):
        for l_index, layer in enumerate(self.layers):
            layer_weights = []

            if l_index == 0:
                for _ in range(0, layer.neuron_amount):
                    neuron_weights = []
                    for _ in range(0, self.input_length):
                        neuron_weights.append(random.random())
                    layer_weights.append(neuron_weights)
            else:
                for _ in range(0, layer.neuron_amount):
                    neuron_weights = []
                    for _ in range(0, self.layers[l_index - 1].neuron_amount):
                        neuron_weights.append(random.random())
                    layer_weights.append(neuron_weights)

            layer.weights = layer_weights

    def create_random_biases(self):
        for _, layer in enumerate(self.layers):
            layer_biases = []
            for neuron in range(layer.neuron_amount):
                layer_biases.append(random.random())
            layer.biases = np.array(layer_biases)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, inputs: np.array):
        self.current_inputs = inputs

        for layer_index, layer in enumerate(self.layers):
            net_inputs = []
            neuron_activations = []

            if layer_index == 0:
                for neuron_index in range(0, layer.neuron_amount):
                    net_input = sum(layer.weights[neuron_index] * self.current_inputs) + layer.biases[neuron_index]
                    net_inputs.append(net_input)

                    activation = self.sigmoid(net_input)
                    neuron_activations.append(activation)
            else:
                previous_layer = self.layers[layer_index-1]

                for neuron_index in range(0, layer.neuron_amount):
                    net_input = sum(layer.weights[neuron_index] * previous_layer.activations) + layer.biases[neuron_index]
                    net_inputs.append(net_input)

                    activation = self.sigmoid(net_input)
                    neuron_activations.append(activation)
            
            layer.net_inputs = np.array(net_inputs)
            layer.activations = np.array(neuron_activations)
            self.current_inputs = layer.activations

            if layer_index == len(self.layers)-1:
                # print(np.argmax(layer.activations))
                # print(layer.activations)
                ...

    def return_output(self):
        return self.layers[-1].activations

    def backwards_pass(self, learning_rate: float, expected_values: np.array):
        reversed_layers = list(reversed(self.layers))

        for layer_index, layer in enumerate(reversed_layers):
            layer.derivatives = layer.activations * (1-layer.activations)
            
            # delta Calculation
            if layer_index == 0:
                layer.delta_values = (expected_values - layer.activations) * layer.derivatives
            else:
                delta_values = []

                for neuron_index in range(layer.neuron_amount):
                    weighted_sum = 0

                    for neuron in range(previous_layer.neuron_amount):
                        weighted_sum += previous_layer.weights[neuron][neuron_index] * previous_layer.delta_values[neuron]
                    
                    delta = weighted_sum * layer.derivatives[neuron_index]
                    delta_values.append(delta)

                layer.delta_values = np.array(delta_values)


            # delta weight calculation and weight change
            next_layer = reversed_layers[layer_index+1]
            for neuron_index in range(layer.neuron_amount):
                for weight_index in range(len(layer.weights[neuron_index])):
                    delta_weight = learning_rate * layer.delta_values[neuron_index] * next_layer.activations[neuron_index]
                    layer.weights[neuron_index][weight_index] += delta_weight

            for neuron_index in range(layer.neuron_amount):
                delta_bias = learning_rate * layer.delta_values[neuron_index]
                layer.biases[neuron_index] += delta_bias
            
            previous_layer = layer
            self.current_inputs = layer.activations

def MSE_Loss(predicted, target):
    return sum((predicted - target)*(predicted - target))

def flatten(matrix):
    new_array = []

    for array in matrix:
        for number in array:
            new_array.append(number)

    return np.array(new_array)

def format_target(target):
    new_array = np.zeros(10)
    new_array[target] = 1
    return np.array(new_array)




mnist = tf.keras.datasets.mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

inputs = np.array(flatten(train_X[0]))

NN = NeuralNetwork(
    [NeuronLayer(4),NeuronLayer(4),NeuronLayer(10)],
    len(inputs)
)

NN.create_random_weights()
NN.create_random_biases()

print(NN.layers[0].weights)

for i in range(0, 100):
    # PREDICTION NOT A NUMBER. CRITICAL ERROR
    NN.forward_pass(flatten(train_X[i]))
    prediction = NN.layers[len(NN.layers)-1].activations
    NN.backwards_pass(0.005, train_y[i])
    loss = MSE_Loss(predicted=prediction, target=train_y[i])
    # print(loss)

print()
print()
print()
print(NN.layers[0].weights)

# NN.forward_pass(flatten(train_X[5]))
# prediction = max(NN.layers[len(NN.layers)-1].activations)
# number = train_y[5]
# print("Prediciton: ", prediction, "   Actual Number: ", number)


# for i in range(0,100):
#     target = train_y[i]
#     array_traget = format_target(target)

#     print(array_traget, "__", target)


