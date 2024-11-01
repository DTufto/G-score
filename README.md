# G-score: An Energy-Aware Metric for Machine Learning Model Evaluation
### By Darrell Tufto and Cato de Kruif


## How to set up the project

1. Clone the repository
2. Install the dependencies using `poetry install`

## How to run the experiment
1. Make sure you have permissions on your system for pyRAPL to take measurements. On Ubuntu 24.04.1 that can be done with:
```shell
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```
2. Within your poetry virtual environment, run the models with
```shell
python experiment.py
```
The experiment will output a CSV which you can evaluate later.

## How to evaluate your models
The code for plotting the results and evaluating the models can be found in the `results` folder.