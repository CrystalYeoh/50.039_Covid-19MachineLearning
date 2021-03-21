import numpy as np
import time
from PIL import Image
from utils import make_balanced_weights, make_weight_losses

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt

from dataset import Lung_Train_Dataset, Lung_Test_Dataset, Lung_Val_Dataset
from utils import make_balanced_weights, make_weight_losses, save
from visualise import plot_curve, visualise_val_predictions

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
        self.model_name = "PreliminaryModel"

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

# Additional fc layer for classification
class AdditionalModel(nn.Module):
    def __init__(self):
        super(AdditionalModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        # self.bnorm = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(5184, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.model_name = "AdditionalModel"

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
        x = self.fc2(x)
        # x = x.view((x.size(0),-1))
        # output = F.log_softmax(x, dim = 1)
        return x

class OverfitModel(nn.Module):
    def __init__(self):
        super(OverfitModel, self).__init__()
        self.model_name = 'OverfitModel'
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1)
        self.maxpool3 = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

#Preliminary Model of 2 Convolutional Layers
class OneModel(nn.Module):
    def __init__(self):
        super(OneModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(21904, 2)
        self.model_name = "OneModel"

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
        self.model_name = "LowPass"
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
        self.model_name = "HighPass"
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

def train(infect_trainloader, infect_testloader, covid_trainloader, covid_testloader, epochs, lr=0.001, weight_decay=0):

    #We first train on infected
    print("Training Binary Classifier model for Infected")

    #Create model, optimizer and criterion
    model_infect = PreliminaryModel()
    model_infect.lr = lr
    optimizer = optim.Adam(model_infect.parameters(),lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    #Train the model
    train_model(model_infect, optimizer, criterion, infect_trainloader, infect_testloader, epochs)

    #Freeze the parameters in the infection model to train the covid model
    for params in model_infect.parameters():
        params.require_grad = False

    #We then train an new model on covid detection
    print("Training Binary Classifier model for Covid")

    #Create model, optimizer and criterion
    # covid model not good with OverfitModel
    model_covid = PreliminaryModel()
    model_covid.lr = lr
    optimizer2 = optim.Adam(model_covid.parameters(),lr=lr, weight_decay=weight_decay)
    criterion2 = nn.CrossEntropyLoss()

    train_model(model_covid, optimizer2, criterion2, covid_trainloader, covid_testloader, epochs, covid = True)

    return model_infect, model_covid

# determine default value of beta
# default is 2 to favour recall to minimize false negatives
def train_model(model, optimizer, criterion, trainloader, testloader, epochs, covid=None,  threshold=0.5, beta=2, eps=1e-9):

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
            test_loss, test_fbeta, test_accuracy = validation(model, testloader, criterion, covid=covid, beta=beta)

            #Printing losses and fbeta scores
            print("Epoch: {}/{} - ".format(e+1, epochs),
            "Training Loss: {:.3f} - ".format(training_loss/len(trainloader)),
            "Training F{}-score: {:.3f} - ".format(beta, train_fbeta),
            "Test Loss: {:.3f} - ".format(test_loss/len(testloader)),
            "Test F{}-score: {:.3f} - ".format(beta,test_fbeta), 
            "Test Accuracy: {:.3f}".format(test_accuracy))
        
        #Record the respective losses and fbeta scores for plotting later
        training_loss_list.append(training_loss/len(trainloader))
        training_acc_list.append(train_fbeta.cpu())
        test_loss_list.append(test_loss/len(testloader))
        test_acc_list.append(test_fbeta.cpu())

        #Reset loss
        training_loss = 0
        if covid:
            save(model, "model/covid_" + model.model_name + ".pt", e)
        else:
            save(model, "model/infected_" + model.model_name + ".pt", e)
        
        model.train()
    #Visualise data after training is done
    plot_curve(training_loss_list, training_acc_list, test_loss_list, test_acc_list, model.model_name, covid)

def test_overall_model(model_infect, model_covid, validloader, criterion, threshold=0.5):

    if torch.cuda.is_available():
        model_infect = model_infect.cuda()
        model_covid = model_covid.cuda()

    #Initiating variables for loss and fbeta scores
    # beta2 = beta**2
    # true_positives = 0
    # predicted_positives = 0
    # target_positives = 0

    equal_i = 0
    equal_c = 0
    equal = 0
    N = 0
    all_images = torch.tensor([]).cuda()
    
    #Loop through the data in batches
    for batch_idx, (images, target_i, target_c) in enumerate(validloader): 

        #Switch images and labels to GPU
        if torch.cuda.is_available():
                images = images.cuda()
                target_i = target_i.cuda()
                target_c = target_c.cuda()
        
        #Put images through model
        output = model_infect.forward(images)

        ps = torch.exp(output)
        predict_infect = torch.ge(output[:,1], threshold)

        output = model_covid.forward(images)
        ps = torch.exp(output)
        predict_covid = torch.ge(output[:,1], threshold)

        equal_i += (target_i == predict_infect).sum()
        equal_c += (target_c == predict_covid).sum()
        N += images.shape[0]
        all_images = torch.cat([all_images, images])

        for i in range(len(predict_covid)):
            true_i = target_i[i].cpu().numpy()
            true_c = target_c[i].cpu().numpy()
            pred_i = predict_infect[i].cpu().numpy()
            pred_c = predict_covid[i].cpu().numpy()

            if true_i == 0:
                ground_truth = 'normal'
            else:
                if true_c == 0:
                    ground_truth = 'non-covid'
                else:
                    ground_truth = 'covid'
            
            if not pred_i:
                prediction = 'normal'
            else:
                if not pred_c:
                    prediction = 'non-covid'
                else:
                    prediction = 'covid'
            
            if ground_truth == prediction:
                equal += 1
            # print("Validation",i)
            # print("True Infected Label:", true_i)
            # print("Predicted Infected Label:",pred_i)
            # print("True Covid Label:", true_c)
            # print("Predicted Covid Label", pred_c)
            # print()
    accuracy = equal/N
    print(f"Infected label Accuracy: {equal_i/N}\nCovid label Accuracy: {equal_c/N}")
    print(f"Overall correctly predicted labels: {accuracy}")
    return all_images, target_i, target_c, predict_infect, predict_covid, accuracy

    #     true_positives += (predicted_labels * target).sum()
    #     predicted_positives += predicted_labels.sum()
    #     target_positives += target.sum()
    
    # #Calculate fbeta

    # # precision = TruePositive / (TruePositive + FalsePositive)
    # precision = true_positives.div(predicted_positives.add(eps))
    # # recall = TruePositive / (TruePositive + FalseNegative)
    # recall = true_positives.div(target_positives.add(eps))
    
    # fbeta = torch.mean((precision*recall).
    #     mul(1 + beta2)
    #     .div(precision.mul(beta2) + recall + eps)
    # )
    
    # return test_loss, fbeta

def validation(model, testloader, criterion, covid = None, threshold=0.5, beta=1, eps=1e-9):

    #Initiating variables for loss and fbeta scores
    beta2 = beta**2
    test_loss = 0
    true_positives = 0
    predicted_positives = 0
    target_positives = 0
    training_accuracy = 0
    total = 0
    
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

        equality = (target.data == output.max(dim=1)[1])
        training_accuracy += equality.type(torch.FloatTensor).sum()
        total += output.shape[0]

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

    training_accuracy = training_accuracy/total
    
    return test_loss, fbeta, training_accuracy

# Define function to load model
def load(path):
    cp = torch.load(path)
    
    # Import pre-trained NN model 
    model_name = cp['model_name']
    if model_name == "PreliminaryModel":
        model = PreliminaryModel()
    elif model_name == "OneModel":
        model = OneModel()
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False

    # Add model info 
    model.lr = cp['c_lr']
    model.model_name = cp['model_name']
    model.load_state_dict(cp['state_dict'])

    print(f"model loaded. Previously trained with {cp['num_of_epochs']}")
    
    return model

dataset_dir = './dataset'

train_transforms = transforms.Compose([
    # transforms.ColorJitter(0.25,0.25,0.25),
    transforms.RandomRotation(15),
    transforms.RandomAffine(15, scale=(0.9,1.1)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4824,), (0.2363,)),
    # transforms.ToPILImage(),
])
# img_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4824,), (0.2363,)),
#     transforms.ToPILImage(),
# ])

# train_transforms=None
img_transforms=None

ld_train = Lung_Train_Dataset(dataset_dir, covid = None, transform=train_transforms)
trainloader = DataLoader(ld_train, batch_size = 64, sampler=WeightedRandomSampler(make_balanced_weights(ld_train), len(ld_train)))
# trainloader = DataLoader(ld_train, batch_size = 64, shuffle=True)
ld_test = Lung_Test_Dataset(dataset_dir, covid = None, transform=img_transforms)
testloader = DataLoader(ld_test, batch_size = 64, shuffle=True)

ld_train_c = Lung_Train_Dataset(dataset_dir, covid = True, transform=train_transforms)
trainloader_c = DataLoader(ld_train_c, batch_size = 64, sampler=WeightedRandomSampler(make_balanced_weights(ld_train_c), len(ld_train_c)))
# trainloader_c = DataLoader(ld_train_c, batch_size = 64, shuffle=True)
ld_test_c = Lung_Test_Dataset(dataset_dir, covid = True, transform=img_transforms)
testloader_c = DataLoader(ld_test_c, batch_size = 64, shuffle=True)

model_infect, model_covid = train(trainloader, testloader, trainloader_c, testloader_c,  epochs=5, lr=0.001, weight_decay=1e-4)

# model_infect = load("infected_OneModel.pt")
# model_covid = load("covid_PreliminaryModel.pt")

ld_valid = Lung_Val_Dataset(dataset_dir,covid=None, transform=img_transforms)
validloader = DataLoader(ld_valid,batch_size=64,shuffle=False)
images, target_i, target_c, pred_i, pred_c, acc = test_overall_model(model_infect,model_covid,validloader,nn.CrossEntropyLoss)
# ld_valid_display = Lung_Val_Dataset(dataset_dir, covid=None)
# visualise_val_predictions(ld_valid_display, target_i, target_c, pred_i, pred_c, acc)

# ld_test = Lung_Test_Dataset(dataset_dir, covid = None)
# testloader = DataLoader(ld_test, batch_size = 64, shuffle=True)
# test(model_infect,model_covid,testloader,nn.CrossEntropyLoss)
