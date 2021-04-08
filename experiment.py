import torch
import torchvision.transforms as transforms
import wandb
import yaml
from training import  random_seed, Dataset, train
from alert import Alert

import pathlib
print(pathlib.Path().absolute())

with open('configs/configs.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

random_seed(config['experiment']['random_seed'])

id = config['experiment']['id'],
train_name = 'training' + str(id)
config['experiment']['id'] += 1

with open('configs/configs.yaml', 'w') as file:
    yaml.dump(config, file)

wandb.init(name=train_name, project='alert', entity='lichtundschatten')

transform = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor()])

train_data = Dataset(config['experiment']['filename_img'], config['experiment']['filename_labels'], transform)
test_data = Dataset(config['experiment']['test_filename_img'], config['experiment']['test_filename_labels'], transform)

data_loader = torch.utils.data.DataLoader(train_data, config['experiment']['batch_size'])
test_loader = torch.utils.data.DataLoader(test_data, config['experiment']['batch_size'])

net = Alert(initialization='normal')

train(net, data_loader, test_loader, config['experiment']['step'], config['experiment']['device_name'],
      num_epoch=config['experiment']['num_epochs'])
