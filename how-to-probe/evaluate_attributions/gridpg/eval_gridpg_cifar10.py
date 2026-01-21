#!/usr/bin/env python3
"""
Evaluate GridPG, Compactness, and Complexity scores for CIFAR10 models.
Computes attribution maps using LRP, GradCAM, InputXGradient, GuidedBackprop.

Usage:
    python eval_gridpg_cifar10.py --model byol --grid_size 3
"""
import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, GaussianBlur
from PIL import Image

# Captum imports
from captum.attr import LayerGradCam, InputXGradient, GuidedBackprop, LayerAttribution

# Zennit imports (for LRP)
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.torchvision import ResNetCanonizer
from zennit.image import imsave

# mmpretrain imports
import mmpretrain
from mmpretrain import get_model


# Model configs
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


class GridPGDataset(Dataset):
    """Dataset for GridPG images."""
    
    def __init__(self, data_root, data_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Load file list
        with open(data_file, 'r') as f:
            self.files = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.data_root, fname)
        
        # Parse labels from filename: {sample_idx}_{label0}_{label1}_...png
        parts = fname.replace('.png', '').split('_')
        labels = [int(x) for x in parts[1:]]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(labels)


def compute_gini_index(array):
    """Compute Gini coefficient (compactness measure)."""
    array = np.array(array, dtype=np.float64)
    array[array < 0] = 0
    array = np.abs(array.flatten())
    array += 1e-10
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def get_pg_score(contribution_map, target_idx, grid_size):
    """Compute GridPG score for a specific grid cell."""
    h, w = contribution_map.shape
    assert h == w, "Height should equal width"
    
    contribution_map = torch.from_numpy(contribution_map)
    contribution_map = F.avg_pool2d(contribution_map.unsqueeze(0).unsqueeze(0), 5, padding=2, stride=1)[0, 0]
    contribution_map = contribution_map.numpy()
    
    # Only positive contributions
    contribution_map[contribution_map < 0] = 0
    
    # Get grid cell coordinates
    i, j = target_idx // grid_size, target_idx % grid_size
    cell_size = h // grid_size
    
    cell_sum = np.sum(contribution_map[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size])
    total_sum = np.sum(contribution_map) + 1e-12
    
    return cell_sum / total_sum


def get_shannon_entropy(contribution_map, grid_size):
    """Compute Shannon entropy (complexity measure)."""
    contribution_map = torch.from_numpy(contribution_map)
    contribution_map = F.avg_pool2d(contribution_map.unsqueeze(0).unsqueeze(0), 5, padding=2, stride=1)[0, 0]
    contribution_map = contribution_map.numpy()
    
    contribution_map[contribution_map < 0] = 0
    contribution_map = contribution_map / (np.sum(contribution_map) + 1e-10)
    
    return -np.sum(contribution_map * np.log2(contribution_map + 1e-10))


def main():
    parser = argparse.ArgumentParser(description='Evaluate GridPG for CIFAR10')
    parser.add_argument('--model', type=str, required=True, choices=['byol', 'dino', 'mocov2'])
    parser.add_argument('--grid_size', type=int, default=3, choices=[2, 3])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()

    # Setup paths
    if args.data_root is None:
        args.data_root = f'grid_pg_images_{args.grid_size}x{args.grid_size}_cifar10_{args.model}'
    if args.data_file is None:
        args.data_file = f'grid_pg_images_{args.grid_size}x{args.grid_size}_list.txt'
    if args.output_dir is None:
        args.output_dir = f'output_gridpg_cifar10_{args.model}_{args.grid_size}x{args.grid_size}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model manually to avoid config compatibility issues
    from mmpretrain.models import ResNet_CIFAR
    
    # Load checkpoint
    model_info = MODEL_CONFIGS[args.model]
    print(f"Loading model: {args.model}")
    print(f"  Checkpoint: {model_info['checkpoint']}")
    
    checkpoint = torch.load(model_info['checkpoint'], map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Build backbone
    backbone = ResNet_CIFAR(depth=50, num_stages=4, out_indices=(3,), style='pytorch')
    
    # Build classifier with proper structure for attribution methods
    class SimpleClassifier(nn.Module):
        def __init__(self, backbone, num_classes=10, in_channels=2048):
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
    
    model = SimpleClassifier(backbone, num_classes=10, in_channels=2048)
    
    # Load weights
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_state_dict[key] = value
        elif key.startswith('head.fc.'):
            new_key = key.replace('head.fc.', 'fc.')
            if value.dim() == 4:  # Conv2d weight [out, in, 1, 1] -> Linear [out, in]
                value = value.squeeze(-1).squeeze(-1)
            new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    # Setup normalization
    norm_fn = BatchNormalize(CIFAR10_MEAN, CIFAR10_STD, device=device)
    img_size = 32 * args.grid_size  # 64 for 2x2, 96 for 3x3

    # Setup dataset
    transform = Compose([ToTensor()])
    dataset = GridPGDataset(args.data_root, args.data_file, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Attribution method results
    attr_methods = ['gradcam', 'ixg', 'gbp']
    results = {method: {'pg_scores': [], 'entropy': [], 'gini': []} for method in attr_methods}

    print(f"\nEvaluating {len(dataset)} grid images...")
    
    for sample_idx, (data, targets) in enumerate(loader):
        if sample_idx >= args.max_samples:
            break
            
        data = data.to(device)
        data_norm = norm_fn(data)
        targets = targets[0]  # Remove batch dim

        # --- GradCAM ---
        layer_gc = LayerGradCam(model, model.backbone.layer4[-1].conv3)
        for target_idx in range(len(targets)):
            attr_map = layer_gc.attribute(data_norm, int(targets[target_idx]))
            attr_map = LayerAttribution.interpolate(attr_map, (img_size, img_size), interpolate_mode='bilinear')
            attr_map = attr_map[0, 0].detach().cpu().numpy()
            
            pg = get_pg_score(attr_map.copy(), target_idx, args.grid_size)
            entropy = get_shannon_entropy(attr_map.copy(), args.grid_size)
            gini = compute_gini_index(attr_map.copy())
            
            results['gradcam']['pg_scores'].append(pg)
            results['gradcam']['entropy'].append(entropy)
            results['gradcam']['gini'].append(gini)

        # --- Input x Gradient ---
        ixg = InputXGradient(model)
        for target_idx in range(len(targets)):
            attr_map = ixg.attribute(data_norm, target=int(targets[target_idx]))
            attr_map = attr_map[0].sum(0).detach().cpu().numpy()
            
            pg = get_pg_score(attr_map.copy(), target_idx, args.grid_size)
            entropy = get_shannon_entropy(attr_map.copy(), args.grid_size)
            gini = compute_gini_index(attr_map.copy())
            
            results['ixg']['pg_scores'].append(pg)
            results['ixg']['entropy'].append(entropy)
            results['ixg']['gini'].append(gini)

        # --- Guided Backprop ---
        gbp = GuidedBackprop(model)
        for target_idx in range(len(targets)):
            attr_map = gbp.attribute(data_norm, target=int(targets[target_idx]))
            attr_map = attr_map[0].sum(0).detach().cpu().numpy()
            
            pg = get_pg_score(attr_map.copy(), target_idx, args.grid_size)
            entropy = get_shannon_entropy(attr_map.copy(), args.grid_size)
            gini = compute_gini_index(attr_map.copy())
            
            results['gbp']['pg_scores'].append(pg)
            results['gbp']['entropy'].append(entropy)
            results['gbp']['gini'].append(gini)

        if (sample_idx + 1) % 10 == 0:
            print(f"  Processed {sample_idx + 1}/{min(len(dataset), args.max_samples)} samples")

        torch.cuda.empty_cache()

    # Print and save results
    print("\n" + "="*60)
    print(f"Results for {args.model} ({args.grid_size}x{args.grid_size} grid)")
    print("="*60)
    print(f"{'Method':<15} {'GridPG↑':<12} {'Entropy↓':<12} {'Gini↑':<12}")
    print("-"*60)
    
    for method in attr_methods:
        pg_mean = np.mean(results[method]['pg_scores'])
        entropy_mean = np.mean(results[method]['entropy'])
        gini_mean = np.mean(results[method]['gini'])
        print(f"{method:<15} {pg_mean:<12.4f} {entropy_mean:<12.4f} {gini_mean:<12.4f}")

    # Save results
    results_file = os.path.join(args.output_dir, 'results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
