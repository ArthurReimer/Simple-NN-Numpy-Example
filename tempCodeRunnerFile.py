correct_predictions = 0
total_samples = 1000

for i in range(total_samples):
    NN.forward_pass(flatten(train_X[i]))
    output = NN.return_output()
    prediction = np.argmax(output)

    if prediction == train_y[i]:
        correct_predictions += 1

    NN.backwards_pass(0.0001, format_target(train_y[i]))

    loss = MSE_Loss(predicted=output, target=format_target(train_y[i]))
    print(f"Loss: {loss}, Prediction: {prediction}, Actual: {train_y[i]}")

accuracy = correct_predictions / total_samples
print(f"\nFinal Accuracy after {total_samples} samples: {accuracy * 100:.2f}%")
