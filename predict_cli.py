import argparse
from torch.utils.data import DataLoader

from model import load, test_overall_model
from dataset import Lung_Test_Dataset, Lung_Val_Dataset
from visualise import visualise_val_predictions

parser = argparse.ArgumentParser(description="Load selected models to view performance")
parser.add_argument("data_dir", help="set path to dataset")
parser.add_argument("infect_checkpoint", help="set path to infect model checkpoint")
parser.add_argument("covid_checkpoint", help="set path to covid model checkpoint")
parser.add_argument("--dataset", default='val', help="set to predict on test or val datasets")

args = parser.parse_args()

model_infect = load(args.infect_checkpoint)
model_covid = load(args.covid_checkpoint)

if args.dataset == 'val':
    ld = Lung_Val_Dataset(args.data_dir, covid = None)
    loader = DataLoader(ld, batch_size = 64, shuffle=True)
    ti, tc, pi, pc, metrics = test_overall_model(model_infect, model_covid, loader)
    visualise_val_predictions(ld, ti, tc, pi, pc, metrics)
    print('You can refer to the visualisation of the validation dataset at "validation_display.png"')
    
else:
    ld = Lung_Test_Dataset(args.data_dir, covid = None)
    loader = DataLoader(ld, batch_size = 64, shuffle=True)
    test_overall_model(model_infect, model_covid, loader)
