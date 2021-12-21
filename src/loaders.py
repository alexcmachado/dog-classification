import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import io
import base64


def transform_test(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return img_str


def get_loaders(train_dir, valid_dir, test_dir):
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
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    loader_train = DataLoader(train_dataset, batch_size=20, shuffle=True)
    loader_valid = DataLoader(valid_dataset, batch_size=20)
    loader_test = DataLoader(test_dataset)

    loaders = {
        "train": loader_train,
        "valid": loader_valid,
        "test": loader_test,
    }

    return loaders
