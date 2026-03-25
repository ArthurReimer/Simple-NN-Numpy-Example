# Simple Neural Network
## Infos
This is an example of an NN in Python with just numpy andworks  well with the MNIST test set. Tensorflow is only used for loading the dataset.
> Its supposed to be simple and not meant to be overly performant or fast, altough the speed is pretty decent for such a simple network.

## Test results
### Network
- **Input Layer:** 784 Neurons
- **Hidden Layer 1:** 48 Neurons
- **Hidden Layer 2:** 48 Neurons
- **Output Layer:** 10 Neurons
> 40416 Total Weights

### Output
``` console
[...]
Epoch 1: Loss = 0.0520, Accuracy = 64.7050%, Time = 19.25 sec
Epoch 2: Loss = 0.0259, Accuracy = 84.9433%, Time = 16.73 sec
Epoch 3: Loss = 0.0199, Accuracy = 88.2917%, Time = 16.90 sec
Epoch 4: Loss = 0.0171, Accuracy = 89.7967%, Time = 16.69 sec
Epoch 5: Loss = 0.0154, Accuracy = 90.7567%, Time = 16.62 sec
Epoch 6: Loss = 0.0142, Accuracy = 91.4600%, Time = 16.46 sec
```
As we can see Numpy gives good performance for such a simple Neural Network, even without batching
