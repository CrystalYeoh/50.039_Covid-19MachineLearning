import numpy as np
import matplotlib.pyplot as plt

# from dataset import Lung_Train_Dataset, Lung_Test_Dataset, Lung_Val_Dataset

def chart_dataset_sizes(dataset):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'normal', 'non-covid', 'covid'
    sizes = list(dataset.dataset_numbers.values())

    fig1, ax1 = plt.subplots()
    pathces, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            startangle=90)
    [_.set_fontsize(15) for _ in texts]
    [_.set_fontsize(15) for _ in autotexts]
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set_title(f"Distribution of class labels for {dataset.dataset_type} dataset\nTotal of {sum(dataset.dataset_numbers.values())} images")

    plt.savefig(f'piechart_{dataset.dataset_type}')
    plt.clf()


def visualise_val_predictions(dataset, target_i, target_c, pred_i, pred_c, metrics):
    accuracy, fbetas, precisions, recalls = metrics

    plt.figure(figsize = (10, 10))
    plt.suptitle('Validation set pictures, with predicted and ground truth labels.\nCovid F2-score: {:.3f}\nAccuracy: {:.3f}\n'.format(fbetas[2], accuracy))

    images = []
    for i in range(len(dataset)):
        images.append(dataset[i][0])
    
    for i in range(len(target_i)):
        plt.subplot(5, 5, i+1)

        plt.xticks([], [])
        plt.yticks([], [])

        if target_i[i] == 0:
            ground_truth = 'normal'
        else:
            if target_c[i] == 0:
                ground_truth = 'non-covid'
            else:
                ground_truth = 'covid'

        if pred_i[i] == 0:
            pred_label = 'normal'
        else:
            if pred_c[i] == 0:
                pred_label = 'non-covid'
            else:
                pred_label = 'covid'

        plt.title(f"Ground Truth Label: {ground_truth}\nPredicted Label: {pred_label}", fontdict={"fontsize":9})
        plt.imshow(images[i].squeeze())
    
    plt.tight_layout()
    plt.savefig('validation_display')

# dataset_dir = './dataset'
# ld_train = Lung_Train_Dataset(dataset_dir)
# chart_dataset_sizes(ld_train)

# ld_test = Lung_Test_Dataset(dataset_dir)
# chart_dataset_sizes(ld_test)

# ld_val = Lung_Val_Dataset(dataset_dir)
# chart_dataset_sizes(ld_val)
