import numpy as np
import time
from dataset import Lung_Train_Dataset, Lung_Test_Dataset
from utils import make_balanced_weights, make_weight_losses

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt


#Preliminary Model of 2 Convolutional Layers
class PreliminaryModel(nn.Module):
    def __init__(self):
        super(PreliminaryModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        # self.bnorm = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(5184, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        # x = self.bnorm(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x
class OverfitModel(nn.Module):
    def __init__(self):
        super(OverfitModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(4, 4, 3, 1)
        self.conv4 = nn.Conv2d(4, 4, 3, 1)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4624, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x

#Preliminary Model of 2 Convolutional Layers
class OneModel(nn.Module):
    def __init__(self):
        super(OneModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(21904, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x


#Low Pass filter before Preliminary model
class LowPassModel(nn.Module):
    def __init__(self):
        super(LowPassModel, self).__init__()
        #Creating a convolutional layer with low pass kernel that is not to be trained
        self.low = nn.Conv2d(1, 1, 3, 1, bias=False)
        kernel_lowpass = (torch.ones(3,3)*(1/9)).expand(self.low.weight.size())
        self.low.weight = nn.Parameter(kernel_lowpass,requires_grad= False)

        #Adding convolutional layers and max pool
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4900, 2)

    def forward(self, x):
        #Apply Low pass filter
        x = self.low(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x

#High Pass filter before Preliminary model
class HighPassModel(nn.Module):
    def __init__(self):
        super(HighPassModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.high = nn.Conv2d(1, 1, 3, 1, bias=False)
        kernel_highpass = (torch.ones(3,3)*(-1/9))
        kernel_highpass[1,1] = 8/9
        kernel_highpass = kernel_highpass.expand(self.high.weight.size())
        self.high.weight = nn.Parameter(kernel_highpass,requires_grad= False)
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4900, 2)

    def forward(self, x):
        x = self.high(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x

def train(infect_trainloader, infect_testloader, covid_trainloader, covid_testloader, epochs, weight_decay=0):

    #We first train on infected
    print("Training Binary Classifier model for Infected")

    #Create model, optimizer and criterion
    model_infect = PreliminaryModel()
    optimizer = optim.Adam(model_infect.parameters(), lr=0.001, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(make_weight_losses(infect_trainloader, False))
    
    #Train the model
    train_model(model_infect, optimizer, criterion, infect_trainloader, infect_testloader, epochs)

    #Freeze the parameters in the infection model to train the covid model
    for params in model_infect.parameters():
        params.require_grad = False

    #We then train an new model on covid detection
    print("Training Binary Classifier model for Covid")

    #Create model, optimizer and criterion
    model_covid = PreliminaryModel()
    optimizer2 = optim.Adam(model_covid.parameters(),lr=0.001, weight_decay=weight_decay)
    criterion2 = nn.CrossEntropyLoss(make_weight_losses(covid_trainloader, True))

    train_model(model_covid, optimizer2, criterion2, covid_trainloader, covid_testloader, epochs, covid = True)


# determine default value of beta
# default is 2 to favour recall to minimize false negatives
def train_model(model, optimizer, criterion, trainloader, testloader, epochs, covid=None, threshold=0.5, beta=2, eps=1e-9):

    #Use GPU
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    #Printing model
    print(model)

    #Initializing variables for printing losses and fbeta scores
    training_loss = 0
    training_loss_list = []
    training_acc_list = []
    test_loss_list = []
    test_acc_list = [] 

    #Initializing variables for fbeta
    beta2 = beta**2
    true_positives = 0
    predicted_positives = 0
    target_positives = 0

    #Loop over the number of epochs
    for e in range(epochs):
        model.train()

        #Loop over the batches
        for batch_idx, (images, target_infected_labels, target_covid_labels) in enumerate(trainloader):

            #Use either covid or infection labels depending on which setting
            if covid:
                target = target_covid_labels
            else:
                target = target_infected_labels
            
            #Transfer images and labels to cpu
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            #Training steps
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            #Recording training loss
            training_loss += loss.item()

            # ps = torch.exp(output)
            # equality = (target_infected_labels.data == ps.max(dim=1)[1])
            # training_accuracy += equality.type(torch.FloatTensor).mean()

            #Recording fbeta
            output = torch.exp(output)
            predicted_labels = torch.ge(output[:,1], threshold)

            true_positives += (predicted_labels * target).sum()
            predicted_positives += predicted_labels.sum()
            target_positives += target.sum()

        #After each epoch, we evaluate the model
        model.eval()
            
        #Turn off gradients for evaluation
        with torch.no_grad():
            
            #Calculating fbeta scores

            #precision = TruePositive / (TruePositive + FalsePositive)
            precision = true_positives.div(predicted_positives.add(eps))
            # recall = TruePositive / (TruePositive + FalseNegative)
            recall = true_positives.div(target_positives.add(eps))

            train_fbeta = torch.mean((precision*recall).
                mul(1 + beta2)
                .div(precision.mul(beta2) + recall + eps)
            )

            #Calculating test loss and test fbeta scores on test set
            test_loss, test_fbeta = validation(model, testloader, criterion, covid=covid, beta=beta)

            #Printing losses and fbeta scores
            print("Epoch: {}/{} - ".format(e+1, epochs),
            "Training Loss: {:.3f} - ".format(training_loss/len(trainloader)),
            "Training Fbeta-score: {:.3f} - ".format(train_fbeta),
            "Test Loss: {:.3f} - ".format(test_loss/len(testloader)),
            "Test Fbeta-score: {:.3f}".format(test_fbeta))
        
        #Record the respective losses and fbeta scores for plotting later
        training_loss_list.append(training_loss/len(trainloader))
        training_acc_list.append(train_fbeta.cpu())
        test_loss_list.append(test_loss/len(testloader))
        test_acc_list.append(test_fbeta.cpu())

        #Reset loss
        training_loss = 0

        #Switch back to training
        model.train()
        
    #Visualise data after training is done
    visualisation(training_loss_list,training_acc_list,test_loss_list,test_acc_list,covid)

def validation(model, testloader, criterion, covid = None, threshold=0.5, beta=1, eps=1e-9):

    #Initiating variables for loss and fbeta scores
    beta2 = beta**2
    test_loss = 0
    true_positives = 0
    predicted_positives = 0
    target_positives = 0
    
    #Loop through the data in batches
    for batch_idx, (images, target_infected_labels, target_covid_labels) in enumerate(testloader): 

        #Set labels depending on covid setting
        if covid:
            target = target_covid_labels
        else:
            target = target_infected_labels

        #Switch images and labels to GPU
        if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
        
        #Put images through model
        output = model.forward(images)

        #Record loss
        test_loss += criterion(output, target).item()
        
        #Record fbeta
        output = torch.exp(output)
        predicted_labels = torch.ge(output[:,1], threshold)

        true_positives += (predicted_labels * target).sum()
        predicted_positives += predicted_labels.sum()
        target_positives += target.sum()
    
    #Calculate fbeta

    # precision = TruePositive / (TruePositive + FalsePositive)
    precision = true_positives.div(predicted_positives.add(eps))
    # recall = TruePositive / (TruePositive + FalseNegative)
    recall = true_positives.div(target_positives.add(eps))
    
    fbeta = torch.mean((precision*recall).
        mul(1 + beta2)
        .div(precision.mul(beta2) + recall + eps)
    )
    
    return test_loss, fbeta

def visualisation(trainingloss, trainingacc, testloss, testacc, covid):

    #Plot the graphs
    plt.xlabel("Training Examples")
    plt.ylabel("Loss/Accuracy")
    plt.plot(np.array(trainingloss),'r',label="Training Loss")
    plt.plot(np.array(trainingacc),'orange',label="Training Accuracy")
    plt.plot(np.array(testloss),'g',label = "Test Loss")
    plt.plot(np.array(testacc),'b',label = "Test Accuracy") 
    plt.plot()
    if covid:
        plt.savefig("covid")
    else:
        plt.savefig("infect")
    plt.clf()

dataset_dir = './dataset'

# ld_train = Lung_Train_Dataset(dataset_dir, covid = None)
# trainloader = DataLoader(ld_train, batch_size = 64, shuffle = True)
# ld_test = Lung_Test_Dataset(dataset_dir, covid = None)
# testloader = DataLoader(ld_test, batch_size = 64, shuffle = True)
# ld_train_c = Lung_Train_Dataset(dataset_dir, covid = True)
# trainloader_c = DataLoader(ld_train_c, batch_size = 64, shuffle = True)
# ld_test_c = Lung_Test_Dataset(dataset_dir, covid = True)
# testloader_c = DataLoader(ld_test_c, batch_size = 64, shuffle = True)
# infected_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4824,), (0.2363,)),
#     transforms.ToPILImage(),
# ])
infected_transforms = transforms.Compose([
    transforms.ColorJitter(0.25,0.25,0.25),
    transforms.RandomRotation(10),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4824,), (0.2363,)),
    # transforms.ToPILImage(),
])
# infected_transforms = None

ld_train = Lung_Train_Dataset(dataset_dir, covid = None, transform=infected_transforms)
trainloader = DataLoader(ld_train, batch_size = 64, sampler=WeightedRandomSampler(make_balanced_weights(ld_train), len(ld_train)))
ld_test = Lung_Test_Dataset(dataset_dir, covid = None)
testloader = DataLoader(ld_test, batch_size = 64, shuffle=True)

ld_train_c = Lung_Train_Dataset(dataset_dir, covid = True)
trainloader_c = DataLoader(ld_train_c, batch_size = 64, sampler=WeightedRandomSampler(make_balanced_weights(ld_train_c), len(ld_train_c)))
ld_test_c = Lung_Test_Dataset(dataset_dir, covid = True)
testloader_c = DataLoader(ld_test_c, batch_size = 64, shuffle=True)

train(trainloader, testloader, trainloader_c, testloader_c,  epochs = 10, weight_decay=1e-4)
