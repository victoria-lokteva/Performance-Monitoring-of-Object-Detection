import torch
import torchvision
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


class Alert(nn.Module):
    def __init__(self, mean_max_std, width=48, height=48):
        super().__init__()
        self.mean_pool = torch.nn.AvgPool2d(kernel_size=(width, height))
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(width, height))
        self.mean_max_std = mean_max_std()
        self.fc1 = nn.Linear(in_features=384*3, out_features=384)
        self.bn1 = nn.BatchNorm2d(384)
        self.fc2 = nn.Linear(in_features=384, out_features=96)
        self.bn2 = nn.BatchNorm2d(96)
        self.fc3 = nn.Linear(in_features=96, out_features=12)
        self.bn3 = nn.BatchNorm2d(12)
        self.fc4 = nn.Linear(in_features=12, out_features=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.fc5 = nn.Linear(in_features=1, out_features=1)

    def statistical_pooling(x):
        std = torch.std(x, dim=1)
        return std

    def mean_max_std(x):
        mean = self.mean_pool(x).view(-1)
        maximum = self.max_pool(x).view(-1)
        return torch.cat((mean, maximum, statistical_pooling(x)))

    def forward(self, x):
        x = self.mean_max_std(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.sigmoid(self.fc5(x))
        return x
