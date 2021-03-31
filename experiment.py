import torch
import torchvision.transforms as transforms
from training import  random_seed, Dataset, train
from alert import Alert

random_seed(10)

transforms = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

train_data = Dataset(filename_img, filename_labels, transforms)
test_data = Dataset(ilename_img, filename_labels, transforms)

data_loader = torch.utils.data.DataLoader(train_data, batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size)

net = Alert()

train(net, data_loader, test_loader)
