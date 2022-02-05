from torchvision.models import VGG, vgg11
from torch.nn import Linear


def get_model() -> VGG:
    model = vgg11(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = Linear(4096, 133, bias=True)

    return model
