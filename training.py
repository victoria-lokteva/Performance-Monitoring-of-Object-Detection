import numpy as np
import torch
from PIL import Image


def random_seed(rs=10):
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.backends.cudnn.deterministic = True

class Dataset(torch.utils.data.Dataset):

    def __init__(self, images_file, labels_file, transforms):
        super().__init__()
        self.images_file = images_file
        self.labels_file = labels_file
        self.transform = transforms
        with open(self.images_file, 'r') as f:
            self.roots = f.readlines()
            self.roots = [root[:-1] for root in self.roots]
        with open(self.labels_file, 'r') as f:
            self.labels = f.readlines()
            self.labels = [label[:-1] for label in self.labels]
            self.labels = [float(label) for label in self.labels]

    def __iter__(self, x):
        return iter(x)

    def __len__(self):
        return len(self.roots)

    def __getitem__(self, index):
        root = self.roots[index]
        label = self.labels[index]
        image = Image.open(root)
        image = self.transform(image)
        return image, label


def test(net, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    predictions = []
    test_loader = iter(test_loader)
    with torch.no_grad():
        for idx, image in enumerate(test_loader):
            image = image.to(device)
            pred = net.forward(image)
            predictions.append(pred)
    return predictions

def train(net, data_loader, lr=0.001, num_epoch=20):

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
    train_loader = iter(data_loader)
    net = net.to(device)

    for epoch in range(num_epoch):
        for idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(image)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

        test(net)
    return net
