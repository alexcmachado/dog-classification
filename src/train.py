import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""

    valid_loss_min = np.Inf

    scheduler = OneCycleLR(
        optimizer, max_lr=1e-4, steps_per_epoch=len(loaders["train"]), epochs=n_epochs
    )

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
            scheduler.step()
            if batch_idx + 1 == len(loaders["train"]):
                print(scheduler.get_last_lr())

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders["valid"]):

            if use_cuda:
                data, target = data.cuda(), target.cuda()

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
