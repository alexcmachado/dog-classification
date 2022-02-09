import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Dict, Tuple
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

BATCH_SIZE = 20


def get_transforms() -> Tuple[Compose, ...]:
    """
    Create composition of transforms for train and validation datasets.

    Returns:
      tuple: Compositions for train and validation.
    """
    transf_train = Compose(
        [
            RandomRotation(30),
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transf_valid = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transf_train, transf_valid


def get_loaders(train_dir: str, valid_dir: str) -> Dict[str, DataLoader]:
    """
    Transform datasets and create dictionary with data loaders.

    Args:
      train_dir (str): Directory where train dataset is located.
      valid_dir (str): Directory where validation dataset is located.

    Returns:
      dict: Data loaders for train and validation.
    """
    print("Get train data loader.")

    transf_train, transf_valid = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=transf_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transf_valid)

    loader_train = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    loader_valid = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    loaders = {
        "train": loader_train,
        "valid": loader_valid,
    }

    return loaders
