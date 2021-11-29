import numpy as np


def test(loaders, model, criterion, device):

    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders["test"]):
        # move to GPU
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss += (1 / (batch_idx + 1)) * (loss.data - test_loss)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print("Test Loss: {:.6f}\n".format(test_loss))

    print(
        "\nTest Accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total, correct, total)
    )
