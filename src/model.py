import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        conv1_out_dim,
        conv2_out_dim,
        conv3_out_dim,
        conv_kernel_size,
        pool_kernel_size,
        drop_prob,
        hidden_dim,
        output_dim,
    ):
        super(Net, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(input_dim, conv1_out_dim, conv_kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out_dim)

        self.conv2 = nn.Conv2d(
            conv1_out_dim, conv2_out_dim, conv_kernel_size, padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_out_dim)

        self.conv3 = nn.Conv2d(
            conv2_out_dim, conv3_out_dim, conv_kernel_size, padding=1
        )
        self.bn3 = nn.BatchNorm2d(conv3_out_dim)

        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
