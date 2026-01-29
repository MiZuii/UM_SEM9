#!/usr/bin/env python3
"""
Get confident images from CIFAR100 test set for GridPG evaluation.
Filters images where the probed model predicts correctly with >95% confidence.

Usage:
    python get_confident_images_cifar100.py --model byol
    python get_confident_images_cifar100.py --model dino
    python get_confident_images_cifar100.py --model mocov2
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
from mmpretrain.models import ResNet_CIFAR

# CIFAR100 fine-grained class names (100 classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Model configs and checkpoints
MODEL_CONFIGS = {
    'byol': {
        'checkpoint': '../../outputs/probing_byol_cifar100/epoch_100.pth',
    },
    'dino': {
        'checkpoint': '../../outputs/probing_dino_cifar100/epoch_100.pth',
    },
    'mocov2': {
        'checkpoint': '../../outputs/probing_mocov2_cifar100/epoch_100.pth',
    },
}

# CIFAR100 normalization
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def main():
    parser = argparse.ArgumentParser(description='Get confident images from CIFAR100')
    parser.add_argument('--model', type=str, required=True, choices=['byol', 'dino', 'mocov2'],
                        help='Model to use for filtering')
    parser.add_argument('--confidence', type=float, default=0.90,
                        help='Confidence threshold (default: 0.90 for CIFAR100)')
    parser.add_argument('--max_per_class', type=int, default=20,
                        help='Maximum images per class (default: 20)')
    parser.add_argument('--data_root', type=str, default='../../data/cifar100',
                        help='CIFAR100 data root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for confident images')
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f'confident_images_cifar100_{args.model}'
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint to get state dict
    model_info = MODEL_CONFIGS[args.model]
    print(f"Loading model: {args.model}")
    print(f"  Checkpoint: {model_info['checkpoint']}")
    
    checkpoint = torch.load(model_info['checkpoint'], map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Build backbone
    backbone = ResNet_CIFAR(depth=50, num_stages=4, out_indices=(3,), style='pytorch')
    
    # Build a simple classifier with the backbone
    class SimpleClassifier(nn.Module):
        def __init__(self, backbone, num_classes=100, in_channels=2048):
            super().__init__()
            self.backbone = backbone
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_channels, num_classes)
        
        def forward(self, x):
            x = self.backbone(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleClassifier(backbone, num_classes=100, in_channels=2048)
    
    # Load weights - map from checkpoint keys to our model keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_state_dict[key] = value
        elif key.startswith('head.fc.'):
            # Map head.fc.weight -> fc.weight and squeeze Conv2d weights to Linear
            new_key = key.replace('head.fc.', 'fc.')
            if value.dim() == 4:  # Conv2d weight [out, in, 1, 1] -> Linear [out, in]
                value = value.squeeze(-1).squeeze(-1)
            new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    # Setup data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR100(root=args.data_root, train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    norm_fn = BatchNormalize(CIFAR100_MEAN, CIFAR100_STD, device=device)

    # Track per-class counts
    class_counts = {i: 0 for i in range(100)}
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
    print(f"Classes with samples: {sum(1 for c in class_counts.values() if c > 0)}/100")
    print("\nPer-class distribution:")
    for i, count in class_counts.items():
        if count > 0:  # Only print classes with samples
            print(f"  {CIFAR100_CLASSES[i]}: {count}")
    
    # Print total
    print(f"\nTotal confident images: {sum(class_counts.values())}")


if __name__ == '__main__':
    main()
