import torch
from torch.utils.data import DataLoader

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

