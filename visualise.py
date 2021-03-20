import numpy as np
import matplotlib.pyplot as plt

def plot_curve(trainingloss, trainingacc, testloss, testacc,covid):
    #Plot the graphs
    plt.xlabel("Training Examples")
    plt.ylabel("Loss/Accuracy")
    plt.plot(np.array(trainingloss),'r',label="Training Loss")
    plt.plot(np.array(trainingacc),'orange',label="Training Accuracy")
    plt.plot(np.array(testloss),'g',label = "Test Loss")
    plt.plot(np.array(testacc),'b',label = "Test Accuracy") 
    plt.legend(loc='upper right', frameon=False)
    plt.plot()
    if covid:
        plt.savefig("covid")
    else:
        plt.savefig("infect")
    plt.clf()

def visualise_val_predictions(dataset, target_i, target_c, pred_i, pred_c, accuracy):
    plt.figure(figsize = (10, 10))
    plt.suptitle('Validation set pictures, with predicted and ground truth labels.\nAccuracy: {:.3f}'.format(accuracy))

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