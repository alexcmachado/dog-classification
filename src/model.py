import torchvision.models as models
import torch.nn as nn


def get_model():
    model = models.vgg11(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 133, bias=True)

    return model
