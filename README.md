# UM Explainability Project

Project examining and trying to replicate **DiET** and **how to probe** frameworks.

## Project Structure and files

```
project directory/
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Handles loading C10, C100, HardMNIST
│   └── model_loader.py   # Handles loading R18, R34
│
├── methods/
│   ├── __init__.py
│   ├── gradcam.py        # GradCAM Class
│   ├── ig.py             # Integrated Gradients Class
│   └── diet.py           # DiET Optimization Logic
│
├── models_trainers/
│   ├── __init__.py
│   ├── baseline_trainer.py   # Baseline models trainer
│   └── diet_trainer.py       # DiET models trainer
│
├── how-to-probe/         # TODO
├── experiment_single.py  # Variant 1: Single Datapoint Cross-Exam
├── pngs/                 # Directory with baseline models training results
├── DiET/                 # Directory for original DiET code
├── results/              # Directory with output of the experiments
└── calculate_models.sh   # Caller to train all baselines and diet models at once
                          # (this is just aggregation for commands, train separately c:)
```

## Seting up environemnt

to setup environment run
```
uv init
uv venv
source .venv/bin/activate
```

## Data setup

The ```CIFAR 10``` and ```CIFAR 100``` datasets will download themself on the first use, soo nothing needs to be done to use them. The Hard Mnist dataset however needs to be prepared. First clone the repository with colored mnist ```git@github.com:jayaneetha/colorized-MNIST.git``` into a directory ```project_root/data```. Than use the script ```python ./DiET/hard_mnist.py``` running it from the project root level, this should complete the data setup.

## Running experiments

The code has few different types of scripts. First if you want to train the baseline and diet models check the ```models_trainers``` directory. For example use of the training commands (these ones were used for our results) check out the ```calculate_models.sh```. After you have all the needed weights you can run experiments. First, generate the target dataseamples for some of thee results with the ```select_examples.py``` script. The main explainability experiments, generating explanations and examples, is in the ```experiment_single.py```. To run the overl model accuracy (and other) tests, you can use ```experiment_models_metrics.py```. Lastly, to run the pixel perturbation test fire up the ```experiment_perturbation.py``` script. If you want to try them all at once here is the code:
```bash
python select_examples.py
python experiment_single.py
python experiment_models_metrics.py
python experiment_perturbation.py
```
After you get the results there are few scripts you need to run to generate the final images and other things.
```bash
python print_models_metrics_results.py
python create_result_matrices.py
python visualize_perturbation.py
```
The first script just prints the metrics results. Second one creates comparison matrices between models, explanation methods and datasets. Third one generates perturbation examples and plots.