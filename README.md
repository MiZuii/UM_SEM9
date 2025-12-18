# UM Explainability Project

Project examining **DiET** and **how to probe** frameworks for explainability.


## Project Structure and files

```
project_root/
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Handles loading C10, C100, HardMNIST
│   └── model_loader.py   # Handles loading R18, R34, R50
├── methods/
│   ├── __init__.py
│   ├── gradcam.py        # GradCAM Class
│   ├── ig.py             # Integrated Gradients Class
│   └── diet.py           # DiET Optimization Logic
├── trainers/
│   ├── __init__.py
│   ├── baseline_trainer.py   # Baseline models trainer
│   └── diet_trainer.py       # DiET models trainer
├── experiment_single.py  # Variant 1: Single Datapoint Cross-Exam
├── experiment_batch.py   # Variant 2: Dataset/Subset Cross-Exam
├── pngs/                 # Directory with baseline models training results
├── DiET/                 # Directory for original DiET code
├── how-to-probe/         # Directory for original how-to-probe code
├── results/              # Directory with output of the experiments
├── calculate_models.sh   # Caller to train all baselines at once
├── DiET.pdf              # DiET paper
└── howtoprobe.pdf        # how-to-probe paper
```

## Data setup

The ```CIFAR 10``` and ```CIFAR 100``` datasets will download themself on the first use, soo nothing needs to be done to use them. The Hard Mnist dataset however needs to be prepared. First clone the repository with colored mnist ```git@github.com:jayaneetha/colorized-MNIST.git``` into a directory ```project_root/data```. Than use the script ```python ./DiET/hard_mnist.py``` running it from the project root level, this should complete the data setup.

## Seting up environemnt

to setup environment run
```
uv init
uv venv
source .venv/bin/activate
```

## Running experiments

To run distillation across all posible combinations on single datapoint run ```python ./experiment_single.py```. Batch experimenting is not finished yet c: