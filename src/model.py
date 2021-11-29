import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_out_dim,
        conv_kernel_size,
        pool_kernel_size,
        hidden_dim,
        drop_prob,
        output_dim,
    ):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(input_dim, conv_out_dim, conv_kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(conv_out_dim)
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
