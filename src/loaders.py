import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def get_loaders(
    resize, crop_size, batch_size, train_dir, valid_dir, test_dir, flip_prob, degrees
):
    print("Get train data loader.")

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(degrees),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(flip_prob),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_valid_test = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid_test)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_valid_test)

    train_dataset_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset_batch = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataset_batch = DataLoader(test_dataset, batch_size=batch_size)

    loaders = {
        "train": train_dataset_batch,
        "valid": valid_dataset_batch,
        "test": test_dataset_batch,
    }

    return loaders
