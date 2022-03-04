from argparse import Namespace, ArgumentParser
import os
import torch
from PIL import Image
import io
from torchvision.models import VGG
import base64
from typing import Union
from torch import Tensor

from trainer import Trainer


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

    trainer = Trainer(use_cuda)

    trainer.get_pretrained_model()
    trainer.load_model_from_disk(model_dir)

    trainer.model.eval()

    print("Done loading model.")
    return trainer.model


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

    transform = Trainer.get_transform()

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
      Namespace: Configurations used for training.
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
    Train model using Trainer class and save to disk.

    Args:
      configs (Configs): Configurations used for training.
    """
    print("Use cuda: {}.".format(configs.use_cuda))

    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(configs.use_cuda)
    trainer.get_pretrained_model()
    trainer.get_loaders(
        train_dir=configs.train_dir,
        valid_dir=configs.valid_dir,
    )
    trainer.prepare_training()

    trainer.train(n_epochs=configs.epochs, model_dir=configs.model_dir)


def main() -> None:
    configs = parser_cli()
    run(configs)


if __name__ == "__main__":
    main()
