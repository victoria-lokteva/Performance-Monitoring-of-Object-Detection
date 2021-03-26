import torch
import torchvision
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


def flatten(tensor):
    return tensor.view(tensor.size(0), -1)

def max_pooling(tensor, num_channels=384):
    result = []
    for channel in range(num_channels):
        one_chan_map = tensor[:,:,channel]
        one_chan_map = flatten(one_chan_map)
        maximum = one_chan_map.max()
        result = torch.cat((result, maximum))
    return result

def mean_pooling(tensor, num_channels=384):
    result = []
    for channel in range(num_channels):
        one_chan_map = tensor[:,:,channel]
        one_chan_map = flatten(one_chan_map)
        mean = one_chan_map.mean()
        result = torch.cat((result, mean))
    return result

def statistical_pooling(tensor, num_channels=384):
    result = []
    for channel in range(num_channels):
        one_chan_map = tensor[:,:,channel]
        one_chan_map = flatten(one_chan_map)
        std = one_chan_map.std()
    result = torch.cat((result, std))
    return result

def mean_max_std(img):
    return torch.cat((mean_pooling(img), max_pooling(img), statistical_pooling(img)))

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
        x = F.softmax(self.fc3(x))
        return x
