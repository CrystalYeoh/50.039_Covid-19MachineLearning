import numpy as np
import time
from dataset import Lung_Train_Dataset, Lung_Test_Dataset

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



class PreliminaryModel(nn.Module):
    def __init__(self):
        super(PreliminaryModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
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


def train(infect_trainloader,infect_testloader,covid_trainloader, covid_testloader, epochs):
    model_infect = PreliminaryModel()
    optimizer = optim.Adam(model_infect.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model_infect,optimizer, criterion, infect_trainloader, infect_testloader, epochs)

    for params in model_infect.parameters():
        params.require_grad = False

    model_covid = PreliminaryModel()
    optimizer = optim.Adam(model_covid.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model_covid,optimizer, criterion, covid_trainloader, covid_testloader, epochs, covid = True)




    
#todo set lr
def train_model(model, optimizer, criterion, trainloader,testloader, epochs, covid = None):

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)

    training_loss = 0
    training_accuracy = 0


    for e in range(epochs):
        model.train()
        for batch_idx, (images, target_infected_labels, target_covid_labels) in enumerate(trainloader):
            if covid:
                target = target_covid_labels
            else:
                target = target_infected_labels
            
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model.forward(images)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            ps = torch.exp(output)
            equality = (target.data == ps.max(dim=1)[1])
            training_accuracy += equality.type(torch.FloatTensor).mean()
            
        model.eval()

        with torch.no_grad():
            test_loss, accuracy = validation(model, testloader, criterion, covid)
            print("Epoch: {}/{} - ".format(e+1, epochs),
            "Training Loss: {:.3f} - ".format(training_loss/len(trainloader)),
            "Training Accuracy: {:.3f} - ".format(training_accuracy/len(trainloader)),
            "Test Loss: {:.3f} - ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        
        model.train()
        training_loss = 0
        training_accuracy = 0

def validation(model, testloader, criterion, covid = None):
    test_loss = 0
    accuracy = 0
    
    for batch_idx, (images, target_infected_labels, target_covid_labels) in enumerate(testloader): 
        if covid:
            target = target_covid_labels
        else:
            target = target_infected_labels

        if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
                
        output = model.forward(images)
        test_loss += criterion(output, target).item()
        
        ps = torch.exp(output)
        equality = (target.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    

    return test_loss, accuracy




dataset_dir = './dataset'

ld_train = Lung_Train_Dataset(dataset_dir, covid = None)
trainloader = DataLoader(ld_train, batch_size = 64, shuffle = True)
ld_test = Lung_Test_Dataset(dataset_dir, covid = None)
testloader = DataLoader(ld_test, batch_size = 64, shuffle = True)
ld_train_c = Lung_Train_Dataset(dataset_dir, covid = True)
trainloader_c = DataLoader(ld_train, batch_size = 64, shuffle = True)
ld_test_c = Lung_Test_Dataset(dataset_dir, covid = True)
testloader_c = DataLoader(ld_test, batch_size = 64, shuffle = True)

train(trainloader, testloader, trainloader_c, testloader_c,  epochs = 10)
