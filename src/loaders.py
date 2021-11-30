import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def get_loaders(
    image_size, batch_size, train_dir, valid_dir, test_dir, flip_prob, erase_prob
):
    print("Get train data loader.")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(flip_prob),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.RandomErasing(erase_prob),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_dataset_batch = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset_batch = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataset_batch = DataLoader(test_dataset, batch_size=batch_size)

    loaders = {
        "train": train_dataset_batch,
        "valid": valid_dataset_batch,
        "test": test_dataset_batch,
    }

    return loaders
