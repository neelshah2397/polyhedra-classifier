from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}


#Data Directory
test_data_dir = 'archive/cube'
train_data_dir = 'archive/dice'
train_datasets = {x: datasets.ImageFolder(os.path.join(train_data_dir, x), data_transforms[x]) for x in ['train', 'val']}
test_datasets = {x: datasets.ImageFolder(os.path.join(test_data_dir, x), data_transforms[x]) for x in ['test']}
test_loader = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['test']}
classes = train_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=False)
#num_ftrs = model.fc.in_features #Modifys classifier in resnet18
num_ftrs = model.classifier.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
# num_ftrs = model.classifier[6].in_features #Modifys last classifier for alexnet
# model.classifier[6] = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('models/densenet121_testbw.pth').state_dict()) #densenet121


def test_all():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader['test']:
            images, labels = data
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in test_loader['test']:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on test images: %d %%' % (100*correct/total))

if __name__ == '__main__':
    #test_all()
    with torch.no_grad():
        for data in test_loader['test']:
            images, labels = data
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            print(predicted, labels)