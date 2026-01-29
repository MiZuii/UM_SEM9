#!/usr/bin/env python3
"""
Generate paper-style attribution comparison figures for CIFAR100 models.
Creates grid visualizations showing input images and attribution maps for different methods.

Layout:
- Rows: Different classes (labels on the left)
- Columns: Input | GradCAM | Input×Grad | GuidedBP

Usage:
    python visualize_attributions_cifar100.py --model byol --num_samples 5
    python visualize_attributions_cifar100.py --model all --num_samples 3
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
        'checkpoint': '../../outputs/probing_byol_cifar100/epoch_100.pth',
        'name': 'BYOL',
    },
    'dino': {
        'checkpoint': '../../outputs/probing_dino_cifar100/epoch_100.pth',
        'name': 'DINO',
    },
    'mocov2': {
        'checkpoint': '../../outputs/probing_mocov2_cifar100/epoch_100.pth',
        'name': 'MoCo v2',
    },
}

# CIFAR100 normalization and class names
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

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


def build_model(checkpoint_path, device):
    """Build model and load weights."""
    backbone = ResNet_CIFAR(depth=50, num_stages=4, out_indices=(3,), style='pytorch')
    
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


def create_single_model_figure(model_name, model, sample_images, sample_labels, device, norm_fn, output_path):
    """Create attribution visualization for a single model."""
    n_samples = len(sample_images)
    attr_methods = ['GradCAM', 'Input×Grad', 'GuidedBP']
    n_methods = len(attr_methods)
    
    # Create figure with extra space for labels and colorbar
    fig, axes = plt.subplots(n_samples, 1 + n_methods, figsize=((1 + n_methods) * 2.2 + 1, n_samples * 2))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    # Adjust subplot positions
    plt.subplots_adjust(left=0.18, right=0.85, wspace=0.08, hspace=0.15)
    
    cmap = plt.cm.bwr
    
    for row, (img, label) in enumerate(zip(sample_images, sample_labels)):
        # Add class label on the very left (truncate long names)
        class_name = CIFAR100_CLASSES[label].replace('_', ' ').title()
        if len(class_name) > 12:
            class_name = class_name[:11] + '.'
        axes[row, 0].annotate(class_name, xy=(-0.4, 0.5), xycoords='axes fraction',
                              fontsize=11, fontweight='bold', ha='right', va='center',
                              color='#333333')
        
        # Input image
        axes[row, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Get attributions
        data_norm = norm_fn(img.unsqueeze(0).to(device))
        attrs = get_attribution_maps(model, data_norm, label, device)
        
        # Plot each method
        for col, method in enumerate(attr_methods):
            attr_map = normalize_attr_map(attrs[method])
            im = axes[row, col + 1].imshow(attr_map, cmap=cmap, vmin=-1, vmax=1)
            if row == 0:
                axes[row, col + 1].set_title(method, fontsize=12, fontweight='bold')
            axes[row, col + 1].axis('off')
    
    # Add colorbar on the far right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attribution', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle(f'{MODEL_CONFIGS[model_name]["name"]} Attribution Maps (CIFAR-100)', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {model_name} figure to {output_path}")


def create_comparison_figure(models_data, sample_images, sample_labels, output_path):
    """
    Create paper-style comparison figure with all models.
    
    Layout:
    Rows: Different input images
    Columns: Input | Model1-GradCAM | Model1-IxG | ... | Colorbar
    """
    n_samples = len(sample_images)
    n_models = len(models_data)
    attr_methods = ['GradCAM', 'Input×Grad', 'GuidedBP']
    n_methods = len(attr_methods)
    
    n_cols = 1 + n_models * n_methods  # Input + (models × methods)
    
    # Create figure with extra space for labels and colorbar
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 1.8 + 1.5, n_samples * 1.8))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    # Adjust subplot positions to make room for labels on left and colorbar on right
    plt.subplots_adjust(left=0.14, right=0.88, wspace=0.05, hspace=0.15)
    
    # Create colormap (blue-white-red)
    cmap = plt.cm.bwr
    im = None  # Will hold the last imshow for colorbar
    
    for row, (img, label) in enumerate(zip(sample_images, sample_labels)):
        # Add class label on the very left (as text annotation)
        class_name = CIFAR100_CLASSES[label].replace('_', ' ').title()
        if len(class_name) > 12:
            class_name = class_name[:11] + '.'
        axes[row, 0].annotate(class_name, xy=(-0.35, 0.5), xycoords='axes fraction',
                              fontsize=10, fontweight='bold', ha='right', va='center',
                              color='#333333')
        
        # Input image
        axes[row, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[row, 0].set_title('Input' if row == 0 else '', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Attribution maps for each model
        col = 1
        for model_name, attrs_list in models_data.items():
            attrs = attrs_list[row]
            for method in attr_methods:
                attr_map = normalize_attr_map(attrs[method])
                im = axes[row, col].imshow(attr_map, cmap=cmap, vmin=-1, vmax=1)
                if row == 0:
                    axes[row, col].set_title(f'{MODEL_CONFIGS[model_name]["name"]}\n{method}', fontsize=9)
                axes[row, col].axis('off')
                col += 1
    
    # Add colorbar on the far right
    if im is not None:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Attribution', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle('Attribution Comparison (CIFAR-100)', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attribution maps for CIFAR100')
    parser.add_argument('--model', type=str, default='all', choices=['byol', 'dino', 'mocov2', 'all'])
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample images')
    parser.add_argument('--data_root', type=str, default='../../data/cifar100')
    parser.add_argument('--output_dir', type=str, default='attribution_figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    norm_fn = BatchNormalize(CIFAR100_MEAN, CIFAR100_STD, device=device)
    
    # Load CIFAR100 test data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=args.data_root, train=False, transform=transform, download=True)
    
    # Select diverse samples from recognizable classes
    # Prefer classes that are easy to visualize: animals, vehicles, objects
    preferred_classes = [3, 8, 15, 19, 31, 48, 58, 69, 88, 90]  # bear, bicycle, camel, cattle, elephant, motorcycle, pickup_truck, rocket, tiger, train
    
    sample_indices = []
    for class_id in preferred_classes:
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        if class_indices:
            sample_indices.append(np.random.choice(class_indices))
        if len(sample_indices) >= args.num_samples:
            break
    
    # If not enough preferred classes, add more random ones
    if len(sample_indices) < args.num_samples:
        for class_id in range(100):
            if class_id in preferred_classes:
                continue
            class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
            if class_indices:
                sample_indices.append(np.random.choice(class_indices))
            if len(sample_indices) >= args.num_samples:
                break
    
    sample_images = [dataset[i][0] for i in sample_indices]
    sample_labels = [dataset[i][1] for i in sample_indices]
    
    print(f"Selected {len(sample_images)} samples from classes: {[CIFAR100_CLASSES[l] for l in sample_labels]}")
    
    # Determine which models to visualize
    if args.model == 'all':
        model_names = ['byol', 'dino', 'mocov2']
    else:
        model_names = [args.model]
    
    # Generate per-model figures
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        model = build_model(MODEL_CONFIGS[model_name]['checkpoint'], device)
        output_path = os.path.join(args.output_dir, f'{model_name}_attributions_cifar100.png')
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
        
        output_path = os.path.join(args.output_dir, 'attribution_comparison_cifar100.png')
        create_comparison_figure(models_data, sample_images, sample_labels, output_path)
    
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
