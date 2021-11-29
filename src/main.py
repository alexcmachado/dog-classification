import argparse
import json
import os
import torch
import timeit

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
    input_dim,
    conv_out_dim,
    conv_kernel_size,
    pool_kernel_size,
    hidden_dim,
    drop_prob,
    output_dim,
    model_dir,
    epochs,
    device,
):

    print("Using device {}.".format(device))

    torch.manual_seed(seed)

    # Load the training data.
    loaders = get_loaders(image_size, batch_size, train_dir, valid_dir, test_dir)

    # Build the model.
    model = Net(
        input_dim=input_dim,
        conv_out_dim=conv_out_dim,
        conv_kernel_size=conv_kernel_size,
        pool_kernel_size=pool_kernel_size,
        hidden_dim=hidden_dim,
        drop_prob=drop_prob,
        output_dim=output_dim,
    ).to(device)

    print("Model loaded")

    # Train the model.
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    model_path = os.path.join(model_dir, "model.pth")

    start_time = timeit.default_timer()

    train(epochs, loaders, model, optimizer, loss_fn, device, model_path)

    end_time = timeit.default_timer()
    t_sec = end_time - start_time
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(
        "Total training time: {:.0f} h, {:.0f} min, {:.0f} sec".format(
            t_hour, t_min, t_sec
        )
    )


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
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    # Model Parameters
    parser.add_argument(
        "--input-dim",
        type=int,
        default=3,
        metavar="N",
        help="size of the input dimension (default: 3)",
    )
    parser.add_argument(
        "--conv-out-dim",
        type=int,
        default=32,
        metavar="N",
        help="size of the convolutional layer output dimension (default: 32)",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=3,
        metavar="N",
        help="size of the convolutional layer kernel (default: 3)",
    )
    parser.add_argument(
        "--pool-kernel-size",
        type=int,
        default=3,
        metavar="N",
        help="size of the pooling layer kernel (default: 3)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        metavar="N",
        help="size of the hidden dimension (default: 2048)",
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=0.5,
        metavar="N",
        help="Drop probability (default: 0.5)",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=133,
        metavar="N",
        help="size of the output dimension (default: 133)",
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
        args.input_dim,
        args.conv_out_dim,
        args.conv_kernel_size,
        args.pool_kernel_size,
        args.hidden_dim,
        args.drop_prob,
        args.output_dim,
        args.model_dir,
        args.epochs,
        device,
    )
