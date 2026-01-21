#!/bin/bash
# Linear probing script for CIFAR10 pretrained SSL models
# Usage: bash utils/run_probing_cifar10.sh [byol|dino|mocov2]

set -e

MODEL=${1:-byol}
NUM_GPUS=${2:-2}
PORT=${3:-29500}

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set paths
MMPRETRAIN_DIR="${PROJECT_ROOT}/pretraining/mmpretrain"
PROBING_DIR="${PROJECT_ROOT}/probing/single_label_classification"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# Select config based on model
case $MODEL in
    byol)
        CONFIG="${PROBING_DIR}/byol_resnet50_linear-probe_cifar10.py"
        WORK_DIR="${OUTPUT_DIR}/probing_byol_cifar10"
        ;;
    dino)
        CONFIG="${PROBING_DIR}/dino_resnet50_linear-probe_cifar10.py"
        WORK_DIR="${OUTPUT_DIR}/probing_dino_cifar10"
        ;;
    mocov2|moco)
        CONFIG="${PROBING_DIR}/mocov2_resnet50_linear-probe_cifar10.py"
        WORK_DIR="${OUTPUT_DIR}/probing_mocov2_cifar10"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: bash run_probing_cifar10.sh [byol|dino|mocov2]"
        exit 1
        ;;
esac

echo "============================================="
echo "Linear Probing on CIFAR10"
echo "============================================="
echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "Work Directory: $WORK_DIR"
echo "GPUs: $NUM_GPUS"
echo "============================================="

# Set PYTHONPATH
export PYTHONPATH="${MMPRETRAIN_DIR}:${PYTHONPATH}"

# Change to mmpretrain directory  
cd "$MMPRETRAIN_DIR"

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=0,1 PORT=$PORT bash ./tools/dist_train.sh "$CONFIG" "$NUM_GPUS" \
        --work-dir "$WORK_DIR" \
        --resume auto
else
    python ./tools/train.py "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --resume auto
fi
