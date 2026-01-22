#!/bin/bash
# DINO Pretraining on CIFAR100 with 2x RTX A4000 GPUs
# Downsized configuration for reasonable training time (~2-4 hours)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DINO_DIR="${PROJECT_ROOT}/pretraining/dino"
DATA_DIR="${PROJECT_ROOT}/data/cifar100_imagefolder"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/dino_cifar100"

# Training hyperparameters (downsized for CIFAR100 and 2x A4000)
EPOCHS=50
BATCH_SIZE_PER_GPU=128
NUM_WORKERS=4
LR=0.003

# CIFAR100-specific crop sizes (32x32 images)
GLOBAL_CROP_SIZE=32
LOCAL_CROP_SIZE=14
GLOBAL_CROPS_SCALE="0.4 1.0"
LOCAL_CROPS_SCALE="0.05 0.4"
LOCAL_CROPS_NUMBER=4

# Architecture
ARCH="resnet50"  # Options: resnet50, bcosresnet50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE_PER_GPU="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bcos)
            ARCH="bcosresnet50"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Setup Python path
export PYTHONPATH="${DINO_DIR}:${PYTHONPATH}"

echo "============================================="
echo "DINO Training on CIFAR100"
echo "============================================="
echo "Architecture: ${ARCH}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "============================================="

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    echo "Please run: python ${SCRIPT_DIR}/download_cifar100.py --output ${DATA_DIR}"
    exit 1
fi

# Determine GPU configuration
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected ${NUM_GPUS} GPU(s)"

# Set B-cos specific options
EXTRA_ARGS=""
if [[ "$ARCH" == "bcosresnet50" ]]; then
    EXTRA_ARGS="--use_bcos_head 1 --optimizer adamw --weight_decay 0.0 --weight_decay_end 0.0"
else
    EXTRA_ARGS="--optimizer sgd --weight_decay 1e-4 --weight_decay_end 1e-4"
fi

# Run training
cd "${DINO_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    torchrun --nproc_per_node=${NUM_GPUS} main_dino.py \
        --arch ${ARCH} \
        --dataset cifar100 \
        --data_path "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --epochs ${EPOCHS} \
        --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
        --num_workers ${NUM_WORKERS} \
        --lr ${LR} \
        --global_crop_size ${GLOBAL_CROP_SIZE} \
        --local_crop_size ${LOCAL_CROP_SIZE} \
        --global_crops_scale ${GLOBAL_CROPS_SCALE} \
        --local_crops_scale ${LOCAL_CROPS_SCALE} \
        --local_crops_number ${LOCAL_CROPS_NUMBER} \
        --warmup_epochs 5 \
        --warmup_teacher_temp_epochs 10 \
        --saveckp_freq 10 \
        ${EXTRA_ARGS}
else
    # Single GPU training
    python main_dino.py \
        --arch ${ARCH} \
        --dataset cifar100 \
        --data_path "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --epochs ${EPOCHS} \
        --batch_size_per_gpu ${BATCH_SIZE_PER_GPU} \
        --num_workers ${NUM_WORKERS} \
        --lr ${LR} \
        --global_crop_size ${GLOBAL_CROP_SIZE} \
        --local_crop_size ${LOCAL_CROP_SIZE} \
        --global_crops_scale ${GLOBAL_CROPS_SCALE} \
        --local_crops_scale ${LOCAL_CROPS_SCALE} \
        --local_crops_number ${LOCAL_CROPS_NUMBER} \
        --warmup_epochs 5 \
        --warmup_teacher_temp_epochs 10 \
        --saveckp_freq 10 \
        ${EXTRA_ARGS}
fi

echo "============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "============================================="
