import argparse
import json
import os
import torch
import timeit

from train import train
from loaders import get_loaders
from model_transfer import get_model
from crit_opt import get_loss_opt


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, "model_info.pth")
    with open(model_info_path, "rb") as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    use_cuda = torch.cuda.is_available()

    model = get_model(output_dim=model_info["output_dim"])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    if use_cuda:
        model.cuda()

    model.eval()

    print("Done loading model.")
    return model


def main(
    seed,
    resize,
    crop_size,
    batch_size,
    train_dir,
    valid_dir,
    test_dir,
    flip_prob,
    degrees,
    output_dim,
    model_dir,
    epochs,
    use_cuda,
):

    print("Use cuda: {}.".format(use_cuda))

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the training data.
    loaders = get_loaders(
        resize,
        crop_size,
        degrees,
        flip_prob,
        batch_size,
        train_dir,
        valid_dir,
        test_dir,
    )

    # Build the model.
    model = get_model(output_dim)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    if use_cuda:
        model.cuda()

    print("Model loaded")

    # Train the model.

    criterion, optimizer = get_loss_opt(params_to_update)

    model_path = os.path.join(model_dir, "model.pth")

    start_time = timeit.default_timer()

    model = train(epochs, loaders, model, optimizer, criterion, use_cuda, model_path)

    end_time = timeit.default_timer()
    t_sec = end_time - start_time
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(
        "Total training time: {:.0f} h, {:.0f} min, {:.0f} sec".format(
            t_hour, t_min, t_sec
        )
    )

    model_info_path = os.path.join(model_dir, "model_info.pth")
    with open(model_info_path, "wb") as f:
        model_info = {
            "output_dim": output_dim,
        }
        torch.save(model_info, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument(
        "--resize",
        type=int,
        default=64,
        metavar="N",
        help="size after resize in px (default: 64)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=64,
        metavar="N",
        help="size after crop (default: 64)",
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
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )

    # Model Parameters
    parser.add_argument(
        "--output-dim",
        type=int,
        default=133,
        metavar="N",
        help="size of the output dimension (default: 133)",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        metavar="N",
        help="Flip probability (default: 0.5)",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=30,
        metavar="N",
        help="Rotation in degrees (default: 30)",
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

    use_cuda = torch.cuda.is_available()

    main(
        args.seed,
        args.resize,
        args.crop_size,
        args.batch_size,
        args.train_dir,
        args.valid_dir,
        args.test_dir,
        args.flip_prob,
        args.degrees,
        args.output_dim,
        args.model_dir,
        args.epochs,
        use_cuda,
    )