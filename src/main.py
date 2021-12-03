import argparse
import json
import os
import torch
from torchsummary import summary
import torchvision.models as models
import timeit
import math

from model import Net
from train import train
from loaders import get_loaders


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
    input_dim,
    conv1_out_dim,
    conv2_out_dim,
    conv3_out_dim,
    conv_kernel_size,
    pool_kernel_size,
    drop_prob,
    output_dim,
    model_dir,
    epochs,
    use_cuda,
    use_transfer,
):

    print("Use cuda: {}.".format(use_cuda))

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the training data.
    loaders = get_loaders(
        resize,
        crop_size,
        batch_size,
        train_dir,
        valid_dir,
        test_dir,
        flip_prob,
        degrees,
    )

    hidden_dim = conv3_out_dim * (math.floor(crop_size / pool_kernel_size ** 3)) ** 2

    # Build the model.
    if use_transfer:
        model = models.vgg11(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        model.classifier[6] = torch.nn.Linear(4096, output_dim, bias=True)

        optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    else:
        model = Net(
            input_dim=input_dim,
            conv1_out_dim=conv1_out_dim,
            conv2_out_dim=conv2_out_dim,
            conv3_out_dim=conv3_out_dim,
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
            hidden_dim=hidden_dim,
            drop_prob=drop_prob,
            output_dim=output_dim,
        )

        optimizer = torch.optim.Adam(model.parameters())

    if use_cuda:
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    summary(
        model,
        input_size=(input_dim, crop_size, crop_size),
        device=device,
        batch_size=batch_size,
    )

    print("Model loaded")

    # Train the model.
    loss_fn = torch.nn.CrossEntropyLoss()

    model_path = os.path.join(model_dir, "model.pth")

    start_time = timeit.default_timer()

    model = train(epochs, loaders, model, optimizer, loss_fn, use_cuda, model_path)

    end_time = timeit.default_timer()
    t_sec = end_time - start_time
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(
        "Total training time: {:.0f} h, {:.0f} min, {:.0f} sec".format(
            t_hour, t_min, t_sec
        )
    )

    return model, loaders, optimizer, loss_fn


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
        "--input-dim",
        type=int,
        default=3,
        metavar="N",
        help="size of the input dimension (default: 3)",
    )
    parser.add_argument(
        "--conv1-out-dim",
        type=int,
        default=8,
        metavar="N",
        help="size of the convolutional layer output dimension (default: 32)",
    )
    parser.add_argument(
        "--conv2-out-dim",
        type=int,
        default=16,
        metavar="N",
        help="size of the convolutional layer output dimension (default: 32)",
    )
    parser.add_argument(
        "--conv3-out-dim",
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
        default=2,
        metavar="N",
        help="size of the pooling layer kernel (default: 3)",
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
    parser.add_argument(
        "--use-transfer",
        type=bool,
        default=False,
        metavar="N",
        help="Use transfer learning (default: False)",
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
        args.input_dim,
        args.conv1_out_dim,
        args.conv2_out_dim,
        args.conv3_out_dim,
        args.conv_kernel_size,
        args.pool_kernel_size,
        args.drop_prob,
        args.output_dim,
        args.model_dir,
        args.epochs,
        use_cuda,
        args.use_transfer,
    )
