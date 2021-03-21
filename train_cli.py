import argparse
from torch.utils.data import DataLoader

from model import pretrain, train
from dataset import Lung_Test_Dataset, Lung_Val_Dataset
from visualise import visualise_val_predictions

parser = argparse.ArgumentParser(description="Train selected models and save them")
parser.add_argument("data_dir", help="set path to dataset")
parser.add_argument("infect_model", help="set infect model")
parser.add_argument("covid_model", help="set covid model")
parser.add_argument("--batch_size", type=int, default=64, help="set batch size")
parser.add_argument("--epochs", type=int, default=5, help="set number of epochs to run")
parser.add_argument("--lr", type=int, default=0.001, help="set learning rate")

args = parser.parse_args()

trainloader, testloader, trainloader_c, testloader_c = pretrain(args.data_dir, args.batch_size)
model_infect, model_covid = train(args.infect_model, args.covid_model, trainloader, testloader, trainloader_c, testloader_c, epochs=args.epochs, lr=args.lr, weight_decay=1e-2)
