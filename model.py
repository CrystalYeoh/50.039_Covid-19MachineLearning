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

#Preliminary Model of 2 Convolutional Layers
class PreliminaryModel(nn.Module):
    def __init__(self):
        super(PreliminaryModel, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool_2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(5184, 2)
        self.model_name = "PreliminaryModel"

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = self.maxpool_2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class ThreeModel(nn.Module):
    def __init__(self):
        super(ThreeModel, self).__init__()
        self.model_name = 'ThreeModel'
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(4, 4, 3, 1)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(1156, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
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
        x = self.fc1(x)
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
        return x

def select_model(model_name):
    model_name = model_name.lower()

    if model_name == "PreliminaryModel".lower():
        model = PreliminaryModel()
    elif model_name == "OneModel".lower():
        model = OneModel()
    elif model_name == "ThreeModel".lower():
        model = ThreeModel()
    elif model_name == "HighPass".lower():
        model = HighPassModel()
    elif model_name == "LowPass".lower():
        model = LowPassModel()
    else:
        raise Exception("Please choose a predefined model")
    return model

def train(infect_model, covid_model, infect_trainloader, infect_testloader, covid_trainloader, covid_testloader, epochs, lr=0.001, weight_decay=0):

    #We first train on infected
    print("Training Binary Classifier model for Infected")

    #Create model, optimizer and criterion
    model_infect = select_model(infect_model)
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
    model_covid = select_model(covid_model)
    model_covid.lr = lr
    optimizer2 = optim.Adam(model_covid.parameters(),lr=lr, weight_decay=weight_decay)
    criterion2 = nn.CrossEntropyLoss()

    train_model(model_covid, optimizer2, criterion2, covid_trainloader, covid_testloader, epochs, covid = True)

    return model_infect, model_covid

def plot_curve(trainingloss, trainingacc, testloss, testacc, model_name, covid):
    #Plot the graphs
    plt.xlabel("Training Examples")
    plt.ylabel("Loss/F2-score")
    plt.plot(np.array(trainingloss),'r',label="Training Loss")
    plt.plot(np.array(trainingacc),'orange',label="Training F2-score")
    plt.plot(np.array(testloss),'g',label = "Test Loss")
    plt.plot(np.array(testacc),'b',label = "Test F2-score") 
    plt.legend(loc='upper right', frameon=False)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=1)
    plt.plot()
    if covid:
        plt.savefig(f"images/covid_{model_name}.png")
    else:
        plt.savefig(f"images/infect_{model_name}.png")
    plt.clf()

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

        graph_lists = [training_loss_list, training_acc_list, test_loss_list, test_acc_list]

        #Reset loss
        training_loss = 0
        if covid:
            save(model, "model/covid_" + model.model_name + ".pt", e, graph_lists)
        else:
            save(model, "model/infected_" + model.model_name + ".pt", e, graph_lists)
        
        model.train()
    #Visualise data after training is done
    plot_curve(training_loss_list, training_acc_list, test_loss_list, test_acc_list, model.model_name, covid)

def test_overall_model(model_infect, model_covid, validloader, threshold=0.5, beta=2):

    if torch.cuda.is_available():
        model_infect = model_infect.cuda()
        model_covid = model_covid.cuda()

    #Initiating variables for loss and fbeta scores
    # beta2 = beta**2
    # true_positives = 0
    # predicted_positives = 0
    # target_positives = 0

    equal = 0
    N = 0
    all_images = torch.tensor([]).cuda()
    # rows will be gnd_truth, cols will be pred
    equality_map = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    
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

        N += images.shape[0]
        all_images = torch.cat([all_images, images])

        for i in range(len(predict_covid)):
            true_i = target_i[i].cpu().numpy()
            true_c = target_c[i].cpu().numpy()
            pred_i = predict_infect[i].cpu().numpy()
            pred_c = predict_covid[i].cpu().numpy()

            if true_i == 0:
                ground_truth = 0
            else:
                if true_c == 0:
                    ground_truth = 1
                else:
                    ground_truth = 2
            
            if not pred_i:
                prediction = 0
            else:
                if not pred_c:
                    prediction = 1
                else:
                    prediction = 2
            
            equality_map[ground_truth][prediction] += 1

    equality_map = np.asarray(equality_map)
    accuracy = 0
    beta2 = beta**2
    precisions = []
    recalls = []
    fbetas = []
    eps = 1e-8

    for c in range(3):
        accuracy += equality_map[c, c]

        true_positives = equality_map[c, c]
        predicted_positives = sum(equality_map[:, c])
        target_positives = sum(equality_map[c, :])
        precision = true_positives/(predicted_positives+eps)
        recall = true_positives/(target_positives+eps)
        fbeta = (1+beta2)*precision*recall/(beta2*precision+recall+eps)

        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(fbeta)

    accuracy = accuracy/N
    metrics = (accuracy, fbetas, precisions, recalls)

    print("Overall correctly predicted labels: {:.3f}".format(accuracy))
    print("F2-scores for: normal {:.3f}, non-covid {:.3f}, covid {:.3f}".format(fbetas[0], fbetas[1], fbetas[2]))
    return target_i, target_c, predict_infect, predict_covid, metrics

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
    model = select_model(model_name)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False

    # Add model info 
    model.lr = cp['c_lr']
    model.model_name = cp['model_name']
    model.load_state_dict(cp['state_dict'])

    # print(f"model loaded. Previously trained with {cp['num_of_epochs']}")
    
    return model

def pretrain(dataset_dir, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(15, scale=(0.9,1.1)),
    ])
    ld_train = Lung_Train_Dataset(dataset_dir, covid = None, transform=train_transforms)
    trainloader = DataLoader(ld_train, batch_size = batch_size, sampler=WeightedRandomSampler(make_balanced_weights(ld_train), len(ld_train)))
    ld_test = Lung_Test_Dataset(dataset_dir, covid = None, transform=None)
    testloader = DataLoader(ld_test, batch_size = batch_size, shuffle=True)

    ld_train_c = Lung_Train_Dataset(dataset_dir, covid = True, transform=train_transforms)
    trainloader_c = DataLoader(ld_train_c, batch_size = batch_size, sampler=WeightedRandomSampler(make_balanced_weights(ld_train_c), len(ld_train_c)))
    ld_test_c = Lung_Test_Dataset(dataset_dir, covid = True, transform=None)
    testloader_c = DataLoader(ld_test_c, batch_size = batch_size, shuffle=True)

    return trainloader, testloader, trainloader_c, testloader_c

