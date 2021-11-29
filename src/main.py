import argparse
import json
import os
import torch

from model import Net
from train import train
from loaders import get_loaders


def main(
    seed,
    image_size,
    batch_size,
    train_dir,
    valid_dir,
    test_dir,
    model_dir,
    epochs,
    device,
):

    print("Using device {}.".format(device))

    torch.manual_seed(seed)

    # Load the training data.
    loaders = get_loaders(image_size, batch_size, train_dir, valid_dir, test_dir)

    # Build the model.
    model = Net().to(device)

    print("Model loaded")

    # Train the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    model_path = os.path.join(model_dir, "model.pth")

    train(epochs, loaders, model, optimizer, loss_fn, device, model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        metavar="N",
        help="size of input images (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    # SageMaker Parameters
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--valid-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )
    parser.add_argument(
        "--test-dir", type=str, default=os.environ["SM_CHANNEL_TESTING"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(
        args.seed,
        args.image_size,
        args.batch_size,
        args.train_dir,
        args.valid_dir,
        args.test_dir,
        args.model_dir,
        args.epochs,
        device,
    )
