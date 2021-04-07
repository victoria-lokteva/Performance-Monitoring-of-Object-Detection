import torch
from torch import nn as nn
import torch.nn.functional as F


class Alert(nn.Module):
    def __init__(self, width=48, height=48, initialization=None, num_channels=384):
        super().__init__()
        if initialization == 'normal':
            initialize_weights = nn.init.xavier_normal
        elif initialization == 'uniform':
            initialize_weights = nn.init.xavier_uniform
        elif initialization is None:
            pass
        else:
            raise Exception('There is no such initialization')

        self.mean_pool = torch.nn.AvgPool2d(kernel_size=(width, height))
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(width, height))
        self.fc1 = nn.Linear(in_features=num_channels*3, out_features=num_channels)

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.fc2 = nn.Linear(in_features=num_channels, out_features=num_channels//4)

        self.bn2 = nn.BatchNorm1d(num_channels//4)
        self.fc3 = nn.Linear(in_features=num_channels//4, out_features=12)

        self.bn3 = nn.BatchNorm1d(12)
        self.fc4 = nn.Linear(in_features=12, out_features=1)

        self.bn4 = nn.BatchNorm1d(1)
        self.fc5 = nn.Linear(in_features=1, out_features=1)

        if initialization is not None:
            initialize_weights(self.fc1.weight)
            initialize_weights(self.fc2.weight)
            initialize_weights(self.fc3.weight)
            initialize_weights(self.fc4.weight)
            initialize_weights(self.fc5.weight)

    def statistical_pooling(self, x):
        std = torch.std(x,dim=(2,3))  # (B*C*H*W) -> (B*C)
        return std

    def mean_max_std(self, x):
        mean = self.mean_pool(x) # -> (B*C*1*1)
        mean = mean.squeeze() # -> (B*C)
        maximum = self.max_pool(x)
        maximum = maximum.squeeze()
        std = self.statistical_pooling(x)
        return torch.cat((mean, maximum, std), dim=1) #-> (B*(3*C))

    def forward(self, x):
        ###
        # x1 is used only for testing:
        x1 = self.mean_max_std(x)
        ###
        x = self.mean_max_std(x)
        ###
        # add extra channels for testing
        for i in range(127):
            x = torch.cat((x,x1), dim=1)
        ###
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

