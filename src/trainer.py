import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop,
)

from torchvision.models import vgg11
from torch.nn import Linear
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 20


class Trainer:
    """
    Class used to train a convolutional neural network to identify dog breeds.

    Attributes:
      use_cuda (bool, optional): Use GPU Accelerated Computing.
      loaders (dict): Data loaders for train and validation.
      model (VGG): Pretrained VGG model.
      params_to_update (list): Model parameters to be updated with training.
      optimizer (Optimizer): Optimizer to use on training.
      criterion (CrossEntropyLoss): Criterion to use on training.
    """

    def __init__(self, use_cuda: bool = False) -> None:
        """
        Construct a Trainer object.

        Args:
        use_cuda (bool, optional): Use GPU Accelerated Computing.
        """
        self.use_cuda = use_cuda

        self.loaders = {}
        self.model = None
        self.params_to_update = []
        self.optimizer = None
        self.criterion = None

    @staticmethod
    def get_transform(use_augmentation: bool = False) -> Compose:
        """
        Create composition of transforms for datasets.

        Args:
          use_augmentation (bool): Use data augmentation on transforms.

        Returns:
          Compose: Composition of transforms.
        """
        if use_augmentation:
            transform = Compose(
                [
                    RandomRotation(30),
                    RandomResizedCrop(224),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = Compose(
                [
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        return transform

    def get_loaders(self, train_dir: str, valid_dir: str) -> None:
        """
        Transform datasets and create dictionary with data loaders.

        Args:
          train_dir (str): Directory where train dataset is located.
          valid_dir (str): Directory where validation dataset is located.
        """
        transf_train = Trainer.get_transform(use_augmentation=True)
        transf_valid = Trainer.get_transform()

        train_dataset = datasets.ImageFolder(train_dir, transform=transf_train)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=transf_valid)

        loader_train = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        loader_valid = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

        self.loaders = {"train": loader_train, "valid": loader_valid}

    def get_pretrained_model(self) -> None:
        """Download pretrained VGG model and change output size to 133."""
        self.model = vgg11(pretrained=True)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[6] = Linear(4096, 133, bias=True)

        self.params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_to_update.append(param)

        if self.use_cuda:
            self.model.cuda()
        print("Model loaded")

    def get_optimizer(self) -> None:
        """Create SGD optimizer for params_to_update"""
        self.optimizer = torch.optim.SGD(
            params=self.params_to_update,
            lr=0.001,
            momentum=0.9,
        )

    def get_criterion(self) -> None:
        """Create criterion to use on training"""
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, n_epochs: int, model_dir: str) -> None:
        """
        Train the model and save to disk.

        Args:
          n_epochs (int): Total number of epochs to train.
          model_dir (str): Directory to save model.
        """

        valid_loss_min = np.Inf

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            train_loss = self.train_model(train_loss)

            valid_loss = self.validate_model(valid_loss)

            print(
                f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}"
            )

            if valid_loss < valid_loss_min:
                self.save_model_to_disk(model_dir)
                valid_loss_min = valid_loss

    def train_model(self, train_loss: float) -> float:
        """
        Train the model and calculate loss.

        Args:
          train_loss (float): Last train loss calculated.

        Returns:
          float: New train loss.
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.loaders["train"]):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)

        return train_loss

    def validate_model(self, valid_loss: float) -> float:
        """
        Validate the model and calculate loss.

        Args:
          valid_loss (float): Last validation loss calculated.

        Returns:
          float: New validation loss.
        """
        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.loaders["valid"]):

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            outputs = self.model(data)
            loss = self.criterion(outputs, target)

            valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

        return valid_loss

    def save_model_to_disk(self, model_dir: str) -> None:
        """
        Save model to disk.

        Args:
          model_dir (str): Directory to save model.
        """
        save_path = os.path.join(model_dir, "model.pth")
        with open(save_path, "wb") as f:
            torch.save(self.model.state_dict(), f)

    def load_model_from_disk(self, model_dir: str) -> None:
        """
        Load model from disk.

        Args:
          model_dir (str): Directory to load model.
        """
        load_path = os.path.join(model_dir, "model.pth")
        with open(load_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
