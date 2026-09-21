import tensorflow as tf
import time
import numpy as np
import network as NN
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

nn = NN.Network()
nn.add_layer(784, 32)
nn.add_layer(32, 16)
nn.add_layer(16, 10)
nn.setup()

epochs = 30
learning_rate = 0.01

losses = np.zeros(epochs)
epochs_arr = np.zeros(epochs)
accurcies = np.zeros(epochs)

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

    losses[e] = avg_loss
    epochs_arr[e] = e
    accurcies[e] = accuracy

    print(f"Epoch {e + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}%, Time = {epoch_duration:.2f} sec")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_arr, losses)
ax1.set(xlabel='Epochs', ylabel='Loss',
        title='MNIST Training: Loss')
ax1.grid()

ax2.plot(epochs_arr, accurcies)
ax2.set(xlabel='Epochs', ylabel='Accuracy (%)',
        title='MNIST Training: Accuracy')
ax2.grid()

fig.tight_layout()

fig.savefig("graph.png")
plt.show()