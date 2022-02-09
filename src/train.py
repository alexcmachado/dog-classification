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


def train_model(
    model: VGG,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: CrossEntropyLoss,
    train_loss: float,
    use_cuda: bool,
):
    """
    Train the model and calculate loss.

    Args:
      model (VGG): Model to train.
      loader (DataLoader): Data loader for train.
      optimizer (Optimizer): Optimizer to use on training.
      criterion (CrossEntropyLoss): Criterion to use on training.
      train_loss (float): Last train loss calculated.
      use_cuda (bool): Use GPU Accelerated Computing.

    Returns:
      model (VGG): Trained model.
      train_loss (float): New train loss.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(loader):

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)

    return model, train_loss


def validate_model(
    model: VGG,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    valid_loss: float,
    use_cuda: bool,
):
    """
    Validate the model and calculate loss.

    Args:
      model (VGG): Model to train.
      loader (DataLoader): Data loader for train.
      criterion (CrossEntropyLoss): Criterion to use on training.
      valid_loss (float): Last validation loss calculated.
      use_cuda (bool): Use GPU Accelerated Computing.

    Returns:
      model (VGG): Trained model.
      valid_loss (float): New validation loss.
    """
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        outputs = model(data)
        loss = criterion(outputs, target)

        valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

    return valid_loss


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

        model = train_model(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            train_loss=train_loss,
            use_cuda=use_cuda,
        )

        validate_model(
            model=model,
            loader=loaders["valid"],
            criterion=criterion,
            valid_loss=valid_loss,
            use_cuda=use_cuda,
        )

        print(
            f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}"
        )

        if valid_loss < valid_loss_min:
            save_model_to_disk(model_dir=model_dir, model=model)
            valid_loss_min = valid_loss
