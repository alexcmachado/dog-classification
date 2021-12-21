import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

BATCH_SIZE = 20


def get_loaders(train_dir, valid_dir):
    print("Get train data loader.")

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid)

    loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    loaders = {
        "train": loader_train,
        "valid": loader_valid,
    }

    return loaders
