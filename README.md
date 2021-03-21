# 50.039_Covid-19MachineLearning

## Quick Start

`train_cli.py`, `predict_cli.py` are the two entry points into our models via the command line interface. 

Run the following to train the infect model and covid model on the OneModel architecture with 5 epochs, batch size of 64, and 0.001 learning rate. 
```
python train_cli.py ./dataset onemodel onemodel --epochs 5 --lr 0.001 --batch_size 64
```

Run the following to use the previously trained models to predict and evaluate the validation set. 
```
python predict_cli.py ./dataset model/infected_OneModel.pt model/covid_OneModel.pt --dataset val
```

## Models

There are 5 models that we have explored in this project:
- OneModel (1 conv layers, followed by a fully-connected layer)
- PreliminaryModel (2 conv layers, followed by a fully-connected layer)
- ThreeModel (3 conv layers, followed by a fully-connected layer)
- HighPass (starts with a high-pass kernel convolution, 2 conv layers, followed by a fully-connected layer)
- LowPass (starts with a low-pass kernel convolution, 2 conv layers, followed by a fully-connected layer)

Take a look at the report to find out more about our findings from the respective models. 

## train_cli.py
Mandatory args:
1. data_dir: the path to where the dataset is located
2. infect_model: the name of the model to be used as the base architecture for the infect classifier
3. covid_model: the name of the model to be used as the base architecture for the covid classifier

Optional args:
- batch_size: defaults to 64
- epochs: defaults to 5
- lr: learning rate; defaults to 0.001

For all trained models under `model/`, they have been trained with the default optional args as shown above with the exception of `infected_PreliminaryModel_50.pt` and `covid_OneModel_50.pt`. 

## predict_cli.py
Mandatory args:
1. data_dir: the path to where the dataset is located
2. infect_checkpoint: the path to where the saved model for the infect classifer
3. covid_checkpoint: the path to where the saved model for the covid classifer

Typically the path to where the saved model is `model/{infected/covid}_{model_name}.pt`.

Optional args:
- dataset: whether to train it on the validation set or train dataset. If trained on the validation dataset, it will also visualise the 25 images. 

## Final Model
To view the results for the final train model, you can run:

```
python predict_cli.py ./dataset model/infected_PreliminaryModel_50.pt model/covid_OneModel_50.pt --dataset val
```

This final model architecture was trained on the PreliminaryModel for the infect classifer and OneModel for the covid classifer for 50 epochs. 
