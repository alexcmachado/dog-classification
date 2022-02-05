import argparse
import os
import torch
import timeit
from PIL import Image
import io
import torchvision.transforms as transforms
import base64

from train import train
from loaders import get_loaders
from model import get_model


class Configs:
    def __init__(self, args):
        self.seed = args.seed
        self.epochs = args.epochs
        self.use_transfer = args.use_transfer
        self.train_dir = args.train_dir
        self.valid_dir = args.valid_dir
        self.model_dir = args.model_dir
        self.use_cuda = args.use_cuda


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

    model = get_model(model_info["use_transfer"])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    if use_cuda:
        model.cuda()

    model.eval()

    print("Done loading model.")
    return model


def input_fn(input_data, content_type):
    if type(input_data) == str:
        input_data = base64.b64decode(input_data)

    img = Image.open(io.BytesIO(input_data))

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_t = transform(img)
    batch_t = img_t.unsqueeze(0)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        batch_t = batch_t.cuda()

    return batch_t


def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.

    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn

    Returns: a prediction
    """
    out = model(data)
    index = out.argmax()

    return index


def parser_cli() -> Configs:
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--use-cuda",
        type=bool,
        default=torch.cuda.is_available(),
        help="use GPU Accelerated Computing",
    )

    # Model Parameters
    parser.add_argument(
        "--use-transfer", type=bool, default=False, help="use transfer learning"
    )

    # SageMaker Parameters
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--valid-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )

    args = parser.parse_args()
    return Configs(args)


def write_model_to_disk(path: str, use_transfer: bool) -> None:
    model_info_path = os.path.join(path, "model_info.pth")
    with open(model_info_path, "wb") as f:
        model_info = {"use_transfer": use_transfer}
        torch.save(model_info, f)


def run(configs: Configs) -> None:
    print("Use cuda: {}.".format(configs.use_cuda))

    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the training data.
    loaders = get_loaders(configs.train_dir, configs.valid_dir)

    # Build the model.
    model = get_model(configs.use_transfer)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    if configs.use_cuda:
        model.cuda()

    print("Model loaded")

    # Train the model.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    model_path = os.path.join(configs.model_dir, "model.pth")

    start_time = timeit.default_timer()

    model = train(
        configs.epochs,
        loaders,
        model,
        optimizer,
        criterion,
        configs.use_cuda,
        model_path,
    )

    end_time = timeit.default_timer()
    t_sec = end_time - start_time
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(
        "Total training time: {:.0f} h, {:.0f} min, {:.0f} sec".format(
            t_hour, t_min, t_sec
        )
    )
    write_model_to_disk()


def main() -> None:
    configs = parser_cli()
    run(configs)


if __name__ == "__main__":
    main()
