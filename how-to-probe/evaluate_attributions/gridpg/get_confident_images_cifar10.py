#!/usr/bin/env python3
"""
Get confident images from CIFAR10 test set for GridPG evaluation.
Filters images where the probed model predicts correctly with >95% confidence.

Usage:
    python get_confident_images_cifar10.py --model byol
    python get_confident_images_cifar10.py --model dino
    python get_confident_images_cifar10.py --model mocov2
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# mmpretrain imports
import mmpretrain
from mmpretrain import get_model

# CIFAR10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model configs and checkpoints
MODEL_CONFIGS = {
    'byol': {
        'config': '../../probing/single_label_classification/byol_resnet50_linear-probe_cifar10.py',
        'checkpoint': '../../outputs/probing_byol_cifar10/epoch_100.pth',
    },
    'dino': {
        'config': '../../probing/single_label_classification/dino_resnet50_linear-probe_cifar10.py',
        'checkpoint': '../../outputs/probing_dino_cifar10/epoch_100.pth',
    },
    'mocov2': {
        'config': '../../probing/single_label_classification/mocov2_resnet50_linear-probe_cifar10.py',
        'checkpoint': '../../outputs/probing_mocov2_cifar10/epoch_100.pth',
    },
}

# CIFAR10 normalization
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def main():
    parser = argparse.ArgumentParser(description='Get confident images from CIFAR10')
    parser.add_argument('--model', type=str, required=True, choices=['byol', 'dino', 'mocov2'],
                        help='Model to use for filtering')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence threshold (default: 0.95)')
    parser.add_argument('--max_per_class', type=int, default=100,
                        help='Maximum images per class (default: 100)')
    parser.add_argument('--data_root', type=str, default='../../data/cifar10',
                        help='CIFAR10 data root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for confident images')
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f'confident_images_cifar10_{args.model}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model_info = MODEL_CONFIGS[args.model]
    print(f"Loading model: {args.model}")
    print(f"  Config: {model_info['config']}")
    print(f"  Checkpoint: {model_info['checkpoint']}")
    
    model = get_model(model_info['config'], pretrained=model_info['checkpoint'])
    model.to(device)
    model.eval()
    model.data_preprocessor = None  # Disable built-in preprocessing
    
    for param in model.parameters():
        param.requires_grad = False

    # Setup data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    norm_fn = BatchNormalize(CIFAR10_MEAN, CIFAR10_STD, device=device)

    # Track per-class counts
    class_counts = {i: 0 for i in range(10)}
    sample_index = 0
    correct_count = 0

    print(f"\nFiltering images with confidence >= {args.confidence}")
    print(f"Max per class: {args.max_per_class}")

    for idx, (data, target) in enumerate(loader):
        target_class = target.item()
        
        # Skip if we have enough for this class
        if class_counts[target_class] >= args.max_per_class:
            continue

        # Normalize and predict
        data_norm = norm_fn(data.to(device))
        with torch.no_grad():
            output = model(data_norm)
            probs = nn.Softmax(dim=1)(output)
            pred_class = probs.argmax(1).item()
            confidence = probs.max().item()

        # Check if correct and confident
        if pred_class == target_class and confidence >= args.confidence:
            # Save image
            img_array = (data[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            fname = f"{sample_index}_{target_class}.png"
            img.save(os.path.join(args.output_dir, fname))
            
            class_counts[target_class] += 1
            sample_index += 1
            correct_count += 1

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images, saved {sample_index} confident images")

        # Check if we have enough for all classes
        if all(c >= args.max_per_class for c in class_counts.values()):
            break

    print(f"\nDone! Saved {sample_index} confident images to {args.output_dir}")
    print("Per-class distribution:")
    for i, count in class_counts.items():
        print(f"  {CIFAR10_CLASSES[i]}: {count}")


if __name__ == '__main__':
    main()
