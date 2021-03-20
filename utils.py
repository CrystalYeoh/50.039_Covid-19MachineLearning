import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from dataset import Lung_Train_Dataset

def make_balanced_weights(dataset):
    covid = dataset.covid

    labels = []
    for images, infected_labels, covid_labels in DataLoader(dataset, batch_size=64, shuffle=False):
        if covid:
            labels += covid_labels
        else:
            labels += infected_labels

    N = len(labels)
    ndata= dataset.dataset_numbers
    max_num_normal = ndata[f"{dataset.dataset_type}_normal"]
    max_num_noncovid = ndata[f"{dataset.dataset_type}_non-covid"]
    max_num_covid = ndata[f"{dataset.dataset_type}_covid"]

    if covid:
        class_count = [N/max_num_noncovid, N/max_num_covid]

    else:
        class_count = [N/max_num_normal, N/(max_num_noncovid+max_num_covid)]
    
    weights = [class_count[labels[i]] for i in range(N)]

    return torch.tensor(weights, dtype=torch.float)

def make_weight_losses(dataloader, covid):
    labels = []
    positive_count = 0
    for images, infected_labels, covid_labels in dataloader:
        if covid:
            labels += covid_labels
            positive_count += covid_labels.sum()
        else:
            labels += infected_labels
            positive_count += infected_labels.sum()

    N = len(labels)

    return torch.tensor([N/(N-positive_count), N/positive_count])

# tensor([0.4824, 0.4824, 0.4824]) tensor([0.2363, 0.2363, 0.2363]) for Lung_Train w/o covid
def cal_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        data = data[0]
        
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def save(model, path):
    checkpoint = {
                  'c_lr': model.lr,
                  'state_dict': model.state_dict(),
                  'model_name': model.model_name,
                  }
    torch.save(checkpoint, path)
