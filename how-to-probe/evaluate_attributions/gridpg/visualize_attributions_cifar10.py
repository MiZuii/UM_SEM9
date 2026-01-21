#!/usr/bin/env python3
"""
Generate paper-style attribution comparison figures for CIFAR10 models.
Creates grid visualizations showing input images and attribution maps for different methods.

Usage:
    python visualize_attributions_cifar10.py --model byol --num_samples 5
    python visualize_attributions_cifar10.py --model all --num_samples 3
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Captum imports
from captum.attr import LayerGradCam, InputXGradient, GuidedBackprop, LayerAttribution

# mmpretrain imports
from mmpretrain.models import ResNet_CIFAR


# Model configs
MODEL_CONFIGS = {
    'byol': {
        'checkpoint': '../../outputs/probing_byol_cifar10/epoch_100.pth',
        'name': 'BYOL',
    },
    'dino': {
        'checkpoint': '../../outputs/probing_dino_cifar10/epoch_100.pth',
        'name': 'DINO',
    },
    'mocov2': {
        'checkpoint': '../../outputs/probing_mocov2_cifar10/epoch_100.pth',
        'name': 'MoCo v2',
    },
}

# CIFAR10 normalization and class names
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def build_model(checkpoint_path, device):
    """Build model and load weights."""
    backbone = ResNet_CIFAR(depth=50, num_stages=4, out_indices=(3,), style='pytorch')
    
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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_state_dict[key] = value
        elif key.startswith('head.fc.'):
            new_key = key.replace('head.fc.', 'fc.')
            if value.dim() == 4:
                value = value.squeeze(-1).squeeze(-1)
            new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


def get_attribution_maps(model, data_norm, target, device, img_size=32):
    """Compute attribution maps using different methods."""
    attributions = {}
    
    # GradCAM
    layer_gc = LayerGradCam(model, model.backbone.layer4[-1].conv3)
    attr_map = layer_gc.attribute(data_norm, target)
    attr_map = LayerAttribution.interpolate(attr_map, (img_size, img_size), interpolate_mode='bilinear')
    attributions['GradCAM'] = attr_map[0, 0].detach().cpu().numpy()
    
    # Input x Gradient  
    ixg = InputXGradient(model)
    attr_map = ixg.attribute(data_norm, target=target)
    attributions['Input×Grad'] = attr_map[0].sum(0).detach().cpu().numpy()
    
    # Guided Backprop
    gbp = GuidedBackprop(model)
    attr_map = gbp.attribute(data_norm, target=target)
    attributions['GuidedBP'] = attr_map[0].sum(0).detach().cpu().numpy()
    
    return attributions


def normalize_attr_map(attr_map, percentile=99):
    """Normalize attribution map for visualization."""
    attr_map = attr_map.copy()
    vmax = np.percentile(np.abs(attr_map), percentile)
    attr_map = np.clip(attr_map, -vmax, vmax)
    attr_map = attr_map / (vmax + 1e-10)
    return attr_map


def create_comparison_figure(models_data, sample_images, sample_labels, output_path):
    """
    Create paper-style comparison figure.
    
    Layout:
    Rows: Different input images
    Columns: Input | Model1-GradCAM | Model1-IxG | Model2-GradCAM | Model2-IxG | ...
    """
    n_samples = len(sample_images)
    n_models = len(models_data)
    attr_methods = ['GradCAM', 'Input×Grad', 'GuidedBP']
    n_methods = len(attr_methods)
    
    n_cols = 1 + n_models * n_methods  # Input + (models × methods)
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 1.5, n_samples * 1.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    # Create colormap (blue-white-red)
    cmap = plt.cm.bwr
    
    for row, (img, label) in enumerate(zip(sample_images, sample_labels)):
        # Input image
        axes[row, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=10)
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=9, rotation=0, labelpad=35, va='center')
        
        # Attribution maps for each model
        col = 1
        for model_name, attrs_list in models_data.items():
            attrs = attrs_list[row]
            for method in attr_methods:
                attr_map = normalize_attr_map(attrs[method])
                axes[row, col].imshow(attr_map, cmap=cmap, vmin=-1, vmax=1)
                if row == 0:
                    axes[row, col].set_title(f'{MODEL_CONFIGS[model_name]["name"]}\n{method}', fontsize=8)
                axes[row, col].axis('off')
                col += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison figure to {output_path}")


def create_single_model_figure(model_name, model, sample_images, sample_labels, device, norm_fn, output_path):
    """Create attribution visualization for a single model."""
    n_samples = len(sample_images)
    attr_methods = ['GradCAM', 'Input×Grad', 'GuidedBP']
    n_methods = len(attr_methods)
    
    fig, axes = plt.subplots(n_samples, 1 + n_methods, figsize=((1 + n_methods) * 2, n_samples * 2))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    cmap = plt.cm.bwr
    
    for row, (img, label) in enumerate(zip(sample_images, sample_labels)):
        # Input image
        axes[row, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=12)
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f'{CIFAR10_CLASSES[label]}', fontsize=10, rotation=0, labelpad=40, va='center')
        
        # Get attributions
        data_norm = norm_fn(img.unsqueeze(0).to(device))
        attrs = get_attribution_maps(model, data_norm, label, device)
        
        # Plot each method
        for col, method in enumerate(attr_methods):
            attr_map = normalize_attr_map(attrs[method])
            im = axes[row, col + 1].imshow(attr_map, cmap=cmap, vmin=-1, vmax=1)
            if row == 0:
                axes[row, col + 1].set_title(method, fontsize=12)
            axes[row, col + 1].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attribution')
    
    plt.suptitle(f'{MODEL_CONFIGS[model_name]["name"]} Attribution Maps', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {model_name} figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attribution maps for CIFAR10')
    parser.add_argument('--model', type=str, default='all', choices=['byol', 'dino', 'mocov2', 'all'])
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample images')
    parser.add_argument('--data_root', type=str, default='../../data/cifar10')
    parser.add_argument('--output_dir', type=str, default='attribution_figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    norm_fn = BatchNormalize(CIFAR10_MEAN, CIFAR10_STD, device=device)
    
    # Load CIFAR10 test data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=transform, download=True)
    
    # Select random samples (one per class for diversity)
    sample_indices = []
    for class_id in range(10):
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        if class_indices:
            sample_indices.append(np.random.choice(class_indices))
        if len(sample_indices) >= args.num_samples:
            break
    
    sample_images = [dataset[i][0] for i in sample_indices]
    sample_labels = [dataset[i][1] for i in sample_indices]
    
    print(f"Selected {len(sample_images)} samples from classes: {[CIFAR10_CLASSES[l] for l in sample_labels]}")
    
    # Determine which models to visualize
    if args.model == 'all':
        model_names = ['byol', 'dino', 'mocov2']
    else:
        model_names = [args.model]
    
    # Generate per-model figures
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        model = build_model(MODEL_CONFIGS[model_name]['checkpoint'], device)
        output_path = os.path.join(args.output_dir, f'{model_name}_attributions.png')
        create_single_model_figure(model_name, model, sample_images, sample_labels, device, norm_fn, output_path)
        del model
        torch.cuda.empty_cache()
    
    # Generate comparison figure if multiple models
    if len(model_names) > 1:
        print("\nGenerating comparison figure...")
        models_data = {}
        for model_name in model_names:
            model = build_model(MODEL_CONFIGS[model_name]['checkpoint'], device)
            attrs_list = []
            for img, label in zip(sample_images, sample_labels):
                data_norm = norm_fn(img.unsqueeze(0).to(device))
                attrs = get_attribution_maps(model, data_norm, label, device)
                attrs_list.append(attrs)
            models_data[model_name] = attrs_list
            del model
            torch.cuda.empty_cache()
        
        output_path = os.path.join(args.output_dir, 'attribution_comparison.png')
        create_comparison_figure(models_data, sample_images, sample_labels, output_path)
    
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
