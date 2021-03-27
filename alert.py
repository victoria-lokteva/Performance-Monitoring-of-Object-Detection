import torch
import torchvision
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


def statistical_pooling(tensor):
    std = torch.std(tensor, dim=(0, 1))
    return std

def mean_max_std(tensor, width=48, height=48):
    t = tensor.permute(2,0,1)
    mean_pool = torch.nn.AvgPool2d(kernel_size=(width, height))
    max_pool = torch.nn.MaxPool2d(kernel_size=(width, height))
    mean = mean_pool(t).view(-1)
    maximum = max_pool(t).view(-1)
    return torch.cat((mean, maximum, statistical_pooling(tensor)))

class Alert(nn.Module):
    def __init__(self, mean_max_std, num_channels):
        super().__init__()
        self.mean_max_std = mean_max_std()
        self.fc1 = nn.Linear(in_features=3*num_channels, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.mean_max_std(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x
