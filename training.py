import numpy as np
import torch
from PIL import Image
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import wandb


def random_seed(rs=10):
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.backends.cudnn.deterministic = True


class Dataset(torch.utils.data.Dataset):

    def __init__(self, images_file, labels_file, transform):
        super().__init__()
        self.images_file = images_file
        self.labels_file = labels_file
        self.transform = transform
        with open(self.images_file, 'r') as f:
            self.roots = f.readlines()
            self.roots = [root[:-1] for root in self.roots]
        with open(self.labels_file, 'r') as f:
            self.labels = f.readlines()
            self.labels = [label[:-1] for label in self.labels]
            self.labels = [float(label) for label in self.labels]
            self.labels = torch.Tensor(self.labels)

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


def test(net, test_loader, device, threshold=0.51):
    loss_func = torch.nn.BCELoss()

    predictions = []
    predictions_prob = []
    labels = []
    test_loss = 0

    test_loader = iter(test_loader)
    net = net.eval()

    with torch.no_grad():
        for idx, (image, label) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            label = label.to(device)

            pred = net.forward(image)
            pred_prob = pred
            pred = pred >= threshold
            pred = pred.float()

            loss = loss_func(pred, label.unsqueeze(1))
            test_loss += loss.item()
            predictions.extend(pred.squeeze().tolist())
            predictions_prob.extend(pred_prob.squeeze().tolist())
            labels.extend(label.tolist())

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = 2 * (recall * precision) / (recall + precision)
    rocauc = roc_auc_score(labels, predictions_prob)

    print('Accuracy: %0.3f %% ' % (accuracy),
          'Precision: %0.3f %% ' % (precision),
          'Recall: %0.3f %% ' % (recall),
          'F1: %0.3f %% ' % (f1),
          'RocAUC: %0.3f %% ' % (rocauc)
          )

    return precision, recall, f1, rocauc, test_loss


def train(net, data_loader, test_loader, step, device_name, lr=0.001, num_epoch=20):

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step)
    best = 0

    for epoch in tqdm(range(num_epoch)):
        train_loader = iter(data_loader)
        train_loss = 0

        for idx, (image, label) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(image)
            loss = loss_func(outputs, label.unsqueeze(1))
            # outputs - > tensor[4,1],  label -> [4], label.unsqueze(1) ->[4,1]
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        precision, recall, f1, rocauc, test_loss = test(net, test_loader, device)
        current = f1
        if current > best:
            best = current
            dict = {}
            dict["model"] = net.state_dict()
            dict['optimizer'] = optimizer.state_dict()
            dict['scheduler'] = scheduler.state_dict()
            dict['Precision'] = precision
            dict['Recall'] = recall
            dict['F1'] = f1
            dict['RocAUC'] = rocauc
            torch.save(dict, './model.pt')

        wandb.log({'Epoch': epoch,
                   'Train_loss': train_loss,
                   'Test_loss' : test_loss,
                   'Precision' : precision,
                   'Recall' : recall,
                   'F1' : f1,
                   'RocAUC': rocauc
                   })

        artifact = wandb.Artifact('ml_perceptron', type='model')
        artifact.add_file('./model.pt')
        artifact.save()

    return net
