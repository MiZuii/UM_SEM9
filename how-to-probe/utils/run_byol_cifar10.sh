#!/bin/bash
# BYOL Pretraining on CIFAR10 with 2x RTX A4000 GPUs
# Uses mmpretrain framework

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MMPRETRAIN_DIR="${PROJECT_ROOT}/pretraining/mmpretrain"
CONFIG_FILE="${MMPRETRAIN_DIR}/configs/byol/byol_resnet50_2xb128-coslr-50e_cifar10.py"
WORK_DIR="${PROJECT_ROOT}/outputs/byol_cifar10"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --bcos)
            CONFIG_FILE="${MMPRETRAIN_DIR}/configs/byol/byol_bcosresnet50_2xb128-coslr-50e_cifar10.py"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create work directory
mkdir -p "${WORK_DIR}"

# Setup Python path
export PYTHONPATH="${MMPRETRAIN_DIR}:${PYTHONPATH}"

echo "============================================="
echo "BYOL Training on CIFAR10"
echo "============================================="
echo "Config: ${CONFIG_FILE}"
echo "Work Directory: ${WORK_DIR}"
echo "============================================="

# Check config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Determine GPU configuration
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected ${NUM_GPUS} GPU(s)"

# Run training
cd "${MMPRETRAIN_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash ./tools/dist_train.sh \
        "${CONFIG_FILE}" \
        ${NUM_GPUS} \
        --work-dir "${WORK_DIR}" \
        --resume 'auto'
else
    # Single GPU training
    python tools/train.py \
        "${CONFIG_FILE}" \
        --work-dir "${WORK_DIR}" \
        --resume 'auto'
fi

echo "============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${WORK_DIR}"
echo "============================================="
