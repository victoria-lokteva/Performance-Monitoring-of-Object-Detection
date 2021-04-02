import torch
import torchvision.transforms as transforms
import wandb
import yaml
from training import  random_seed, Dataset, train
from alert import Alert

random_seed(10)

with open('configs.yaml') as file:
    config = yaml.load(file, loader=yaml.FullLoader)

wandb.init(name='training', project='alert', entity='')

transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

train_data = Dataset(filename_img, filename_labels, transform)
test_data = Dataset(test_filename_img, test_filename_labels, transform)

data_loader = torch.utils.data.DataLoader(train_data, batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size)

net = Alert(initialization)

train(net, data_loader, test_loader)
