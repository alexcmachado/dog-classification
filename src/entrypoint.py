from argparse import Namespace, ArgumentParser
import os
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
from torchvision.models import VGG
import base64
from typing import Union
from torch import Tensor

from train import train
from loaders import get_loaders
from model import get_model


def model_fn(model_dir: str) -> VGG:
    """
    Load the PyTorch model from directory.

    Args:
      model_dir (str): Directory to load model.

    Returns:
      VGG: Loaded model.
    """
    print("Loading model.")

    # Determine the device and construct the model.
    use_cuda = torch.cuda.is_available()

    model = get_model()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    if use_cuda:
        model.cuda()

    model.eval()

    print("Done loading model.")
    return model


def input_fn(input_data: Union[str, bytearray], content_type: str) -> Tensor:
    """
    Deserialize input, apply transforms and create batch.
    Move input to GPU, if available.

    Args:
      input_data (str | bytearray): Data to be deserialized.
      content_type (str): Type of input data.

    Returns:
      Tensor: Tensor with input data.
    """
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


def predict_fn(data: Tensor, model: VGG) -> Tensor:
    """
    Call a model on data deserialized in input_fn.

    Args:
      data (Tensor): Input data for prediction deserialized by input_fn.
      model (VGG): PyTorch model loaded in memory by model_fn.

    Returns:
      Tensor: Predicted class.
    """
    out = model(data)
    index = out.argmax()

    return index


def parser_cli() -> Namespace:
    """
    Parse command-line arguments and create Configs instance.

    Returns:
      Configs: Configurations used for training.
    """
    parser = ArgumentParser()

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

    # SageMaker Parameters
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--valid-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )

    args = parser.parse_args()
    return args


def run(configs: Namespace) -> None:
    """
    Create loaders, model, criterion and optimizer, and train the model.

    Args:
      configs (Configs): Configurations used for training.
    """
    print("Use cuda: {}.".format(configs.use_cuda))

    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the training data.
    loaders = get_loaders(train_dir=configs.train_dir, valid_dir=configs.valid_dir)

    # Build the model.
    model = get_model()

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    if configs.use_cuda:
        model.cuda()

    print("Model loaded")

    # Train the model.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    train(
        n_epochs=configs.epochs,
        loaders=loaders,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        use_cuda=configs.use_cuda,
        model_dir=configs.model_dir,
    )


def main() -> None:
    configs = parser_cli()
    run(configs)


if __name__ == "__main__":
    main()
