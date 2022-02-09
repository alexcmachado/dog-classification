import torch
import os
import numpy as np
from torchvision.models import VGG
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from typing import Dict
from torch.utils.data import DataLoader

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_model_to_disk(model_dir: str, model: VGG) -> None:
    """
    Save model to disk.

    Args:
      model_dir (str): Directory to save model.
      model (VGG): Model to save.
    """
    save_path = os.path.join(model_dir, "model.pth")
    with open(save_path, "wb") as f:
        torch.save(model.state_dict(), f)


def train(
    n_epochs: int,
    loaders: Dict[str, DataLoader],
    model: VGG,
    optimizer: Optimizer,
    criterion: CrossEntropyLoss,
    use_cuda: bool,
    model_dir: str,
) -> None:
    """
    Train the model and save to disk.

    Args:
      n_epochs (int): Total number of epochs to train.
      loaders (dict): Data loaders for train and validation.
      model (VGG): Model to train.
      optimizer (Optimizer): Optimizer to use on training.
      criterion (CrossEntropyLoss): Criterion to use on training.
      use_cuda (bool): Use GPU Accelerated Computing.
      model_dir (str): Directory to save model.
    """

    valid_loss_min = np.Inf

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
            save_model_to_disk(model_dir=model_dir, model=model)
            valid_loss_min = valid_loss
