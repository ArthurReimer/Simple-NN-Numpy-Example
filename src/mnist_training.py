import tensorflow as tf
import time
import numpy as np
import network as NN

mnist = tf.keras.datasets.mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

nn = NN.Network()
nn.add_layer(784, 48)
nn.add_layer(48, 48)
nn.add_layer(48, 10)
nn.setup()

epochs = 10
learning_rate = 0.01

for e in range(epochs):
    start_time = time.time()

    correct_predictions = 0
    total_loss = 0

    for i in range(len(train_X)):
        inputs = NN.flatten(train_X[i])
        target = NN.format_target(train_y[i])

        nn.forward_pass(inputs)
        output = nn.return_output()

        prediction = np.argmax(output)
        if prediction == train_y[i]:
            correct_predictions += 1

        nn.backward_pass(inputs, learning_rate, target)

        total_loss += NN.MSE_Loss(output, target)

    epoch_duration = time.time() - start_time
    avg_loss = total_loss / len(train_y)
    accuracy = correct_predictions / len(train_y) * 100

    print(f"Epoch {e + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}%, Time = {epoch_duration:.2f} sec")
