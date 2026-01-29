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
├── how-to-probe/         # SSL Probing Framework (see below)
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

---

# How to Probe

The official implementation of **How to Probe: Simple Yet Effective Techniques for Improved Post-hoc Explanations** @ ICLR 2025 ([OpenReview Link](https://openreview.net/pdf?id=57NfyYxh5f)).

This provides a complete pipeline for evaluating the interpretability of Self-Supervised Learning (SSL) representations on **CIFAR-10** and **CIFAR-100** datasets.

## Pipeline Overview

The evaluation pipeline consists of three stages:

1. **Pre-training**: Train SSL backbones (BYOL, DINO, MoCo v2) on unlabeled data.
2. **Probing**: Train linear classifiers on frozen pretrained features.
3. **Evaluating Attributions**: Use GridPG to measure localization quality of attribution methods.

---

## Quick Start: CIFAR-10

### Step 0: Download Dataset

```bash
# Download and prepare CIFAR-10 in ImageFolder format (required for DINO)
python utils/download_cifar10.py --output_dir data/cifar10
```

### Step 1: Pre-training

Train SSL backbones on CIFAR-10:

```bash
# BYOL
bash utils/run_byol_cifar10.sh

# DINO
bash utils/run_dino_cifar10.sh

# MoCo v2
bash utils/run_moco_cifar10.sh
```

**Checkpoints** will be saved to `work_dirs/` by default.

### Step 2: Linear Probing

Train linear classifiers on frozen backbones:

```bash
bash utils/run_probing_cifar10.sh
```

This trains a linear probe for each pretrained model and saves the classifier weights.

### Step 3: Evaluating Attributions (GridPG)

#### 3.1 Get Confident Images

Filter test images where models make confident, correct predictions:

```bash
cd evaluate_attributions/gridpg

# For each model (byol, dino, mocov2)
python get_confident_images_cifar10.py --model byol --confidence 0.9
python get_confident_images_cifar10.py --model dino --confidence 0.9
python get_confident_images_cifar10.py --model mocov2 --confidence 0.9
```

#### 3.2 Create Grid Dataset

Generate 3×3 grid images from confident samples:

```bash
python create_point_grid_dataset_cifar10.py --model byol
python create_point_grid_dataset_cifar10.py --model dino
python create_point_grid_dataset_cifar10.py --model mocov2
```

#### 3.3 Evaluate GridPG Metrics

Compute GridPG scores, entropy, and Gini index for each attribution method:

```bash
python eval_gridpg_cifar10.py --model byol
python eval_gridpg_cifar10.py --model dino
python eval_gridpg_cifar10.py --model mocov2
```

---

## Quick Start: CIFAR-100

### Step 0: Download Dataset

```bash
python utils/download_cifar100.py --output_dir data/cifar100
```

### Step 1: Pre-training

```bash
# BYOL
bash utils/run_byol_cifar100.sh

# DINO
bash utils/run_dino_cifar100.sh

# MoCo v2
bash utils/run_moco_cifar100.sh
```

### Step 2: Linear Probing

```bash
bash utils/run_probing_cifar100.sh
```

### Step 3: Evaluating Attributions (GridPG)

```bash
cd evaluate_attributions/gridpg

# Get confident images
python get_confident_images_cifar100.py --model byol --confidence 0.9
python get_confident_images_cifar100.py --model dino --confidence 0.9
python get_confident_images_cifar100.py --model mocov2 --confidence 0.9

# Create grid dataset
python create_point_grid_dataset_cifar100.py --model byol
python create_point_grid_dataset_cifar100.py --model dino
python create_point_grid_dataset_cifar100.py --model mocov2

# Evaluate GridPG
python eval_gridpg_cifar100.py --model byol
python eval_gridpg_cifar100.py --model dino
python eval_gridpg_cifar100.py --model mocov2
```

---

## Visualization

Generate attribution comparison figures:

```bash
cd evaluate_attributions/gridpg

# CIFAR-10
python visualize_attributions_cifar10.py --model byol

# CIFAR-100
python visualize_attributions_cifar100.py --model byol

# Probe comparison (Linear vs MLP)
python visualize_probe_comparison.py --model byol
```

---

## How-to-Probe Structure

```
how-to-probe/
├── utils/                          # Utility scripts
│   ├── download_cifar10.py         # Download & convert CIFAR-10
│   ├── download_cifar100.py        # Download & convert CIFAR-100
│   ├── run_byol_cifar10.sh         # BYOL pretraining script
│   ├── run_dino_cifar10.sh         # DINO pretraining script
│   ├── run_moco_cifar10.sh         # MoCo v2 pretraining script
│   ├── run_probing_cifar10.sh      # Linear probing script
│   └── ...                         # CIFAR-100 equivalents
├── pretraining/                    # SSL implementations
│   ├── dino/                       # DINO (Facebook Research)
│   └── mmpretrain_configs/         # BYOL/MoCo v2 configs (OpenMMLab)
├── probing/                        # Linear/MLP probe training
│   └── single_label_classification/
└── evaluate_attributions/          # Attribution evaluation
    ├── gridpg/                     # Grid Pointing Game
    │   ├── get_confident_images_*.py
    │   ├── create_point_grid_dataset_*.py
    │   ├── eval_gridpg_*.py
    │   └── visualize_*.py
    └── epg/                        # Energy Pointing Game
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- mmpretrain (for BYOL/MoCo v2)
- captum (for attribution methods)
- numpy, PIL, matplotlib