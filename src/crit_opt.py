import torch.nn as nn
import torch.optim as optim


def get_loss_opt(parameters):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

    return criterion, optimizer
