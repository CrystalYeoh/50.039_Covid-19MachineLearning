import numpy as np
import time
from dataset import Lung_Train_Dataset

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



class PreliminaryModel(nn.Module):
    def __init__(self):
        super(PreliminaryModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(5184, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        # x = x.view((x.size(0),-1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim = 1)
        return output

    
#todo set lr
def train(trainloader, epochs):
    model = PreliminaryModel()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)

    running_loss = 0

    for e in range(epochs):
        model.train()
        for batch_idx, (images_data, target_infected_labels, target_covid_labels) in enumerate(trainloader):
            
            if torch.cuda.is_available():
                images_data = images_data.cuda()
                target_infected_labels = target_infected_labels.cuda()

            optimizer.zero_grad()
            output = model.forward(images_data)

            loss = criterion(output, target_infected_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print("Training loss:",running_loss/len(trainloader))


dataset_dir = './dataset'

ld_train = Lung_Train_Dataset(dataset_dir)
trainloader = DataLoader(ld_train, batch_size = 4, shuffle = True)

train(trainloader, epochs = 1)
