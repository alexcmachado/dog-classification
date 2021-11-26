import torch
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(n_epochs, loaders, model, optimizer, criterion, device, save_path):
    """returns trained model"""

    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders["valid"]):

            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)

            valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        if valid_loss < valid_loss_min:
            with open(save_path, "wb") as f:
                torch.save(model.state_dict(), f)

            valid_loss_min = valid_loss

    return model
