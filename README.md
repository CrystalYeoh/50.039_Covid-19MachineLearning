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



