#!/usr/bin/env python3
"""
Generate paper-style attribution comparison figures for CIFAR100 models.
Shows Input, GradCAM, Input×Gradient, and GuidedBackprop for BYOL, DINO, MoCo v2.

Usage:
    python visualize_attributions_cifar100.py --model all --num_samples 4
    python visualize_attributions_cifar100.py --model byol --num_samples 6
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Captum imports
from captum.attr import LayerGradCam, InputXGradient, GuidedBackprop, LayerAttribution

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

# Model configs
MODEL_CONFIGS = {
    'byol': {
        'checkpoint': '../../outputs/probing_byol_cifar100/epoch_100.pth',
        'name': 'BYOL'
    },
    'dino': {
        'checkpoint': '../../outputs/probing_dino_cifar100/epoch_100.pth',
        'name': 'DINO'
    },
    'mocov2': {
        'checkpoint': '../../outputs/probing_mocov2_cifar100/epoch_100.pth',
        'name': 'MoCo v2'
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


def build_model(checkpoint_path, device):
    """Build and load a probed model."""
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
    
    # Load weights
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


def get_attributions(model, data_norm, target, img_size=32):
    """Get attribution maps using different methods."""
    attributions = {}
    
    # GradCAM
    layer_gc = LayerGradCam(model, model.backbone.layer4[-1].conv3)
    attr_map = layer_gc.attribute(data_norm, target)
    attr_map = LayerAttribution.interpolate(attr_map, (img_size, img_size), interpolate_mode='bilinear')
    attributions['gradcam'] = attr_map[0, 0].detach().cpu().numpy()
    
    # Input x Gradient
    ixg = InputXGradient(model)
    attr_map = ixg.attribute(data_norm, target=target)
    attributions['ixg'] = attr_map[0].sum(0).detach().cpu().numpy()
    
    # Guided Backprop
    gbp = GuidedBackprop(model)
    attr_map = gbp.attribute(data_norm, target=target)
    attributions['gbp'] = attr_map[0].sum(0).detach().cpu().numpy()
    
    return attributions


def normalize_attr(attr_map, percentile=99):
    """Normalize attribution map for visualization."""
    attr_map = attr_map.copy()
    attr_map = np.maximum(attr_map, 0)
    vmax = np.percentile(attr_map, percentile)
    if vmax > 0:
        attr_map = attr_map / vmax
    return np.clip(attr_map, 0, 1)


def create_comparison_figure(models, sample_data, device, norm_fn, output_path):
    """Create comparison figure across models."""
    model_names = list(models.keys())
    n_models = len(model_names)
    n_samples = len(sample_data)
    n_cols = 4  # Input, GradCAM, IxG, GBP
    
    fig, axes = plt.subplots(n_models * n_samples, n_cols, 
                             figsize=(n_cols * 2.5, n_models * n_samples * 2.5))
    
    plt.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.02, wspace=0.05, hspace=0.15)
    
    # Custom colormap
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('attr', colors)
    
    col_titles = ['Input', 'GradCAM', 'Input×Grad', 'GuidedBP']
    
    row_idx = 0
    for model_name in model_names:
        model = models[model_name]['model']
        display_name = models[model_name]['display_name']
        
        for img, label in sample_data:
            data_norm = norm_fn(img.unsqueeze(0).to(device))
            attrs = get_attributions(model, data_norm, label)
            
            # Input image
            axes[row_idx, 0].imshow(img.permute(1, 2, 0).numpy())
            axes[row_idx, 0].axis('off')
            
            # Attribution maps
            for col, (method, attr_map) in enumerate([('gradcam', attrs['gradcam']), 
                                                       ('ixg', attrs['ixg']),
                                                       ('gbp', attrs['gbp'])], 1):
                attr_norm = normalize_attr(attr_map)
                axes[row_idx, col].imshow(attr_norm, cmap=cmap)
                axes[row_idx, col].axis('off')
            
            # Add class name on left
            class_name = CIFAR100_CLASSES[label][:8]  # Truncate long names
            axes[row_idx, 0].annotate(f'{display_name}\n{class_name}', 
                                       xy=(-0.3, 0.5), xycoords='axes fraction',
                                       fontsize=9, fontweight='bold', ha='right', va='center')
            
            row_idx += 1
    
    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attribution', fontsize=10)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attributions for CIFAR100')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['byol', 'dino', 'mocov2', 'all'])
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='../../data/cifar100')
    parser.add_argument('--output_dir', type=str, default='attribution_figures_cifar100')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    norm_fn = BatchNormalize(CIFAR100_MEAN, CIFAR100_STD, device=device)
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=args.data_root, train=False, transform=transform, download=True)
    
    # Select diverse samples
    sample_indices = []
    np.random.seed(args.seed)
    for class_id in range(0, 100, 100 // args.num_samples):
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
        if class_indices:
            sample_indices.append(np.random.choice(class_indices))
        if len(sample_indices) >= args.num_samples:
            break
    
    sample_data = [(dataset[i][0], dataset[i][1]) for i in sample_indices]
    print(f"Selected samples: {[CIFAR100_CLASSES[l] for _, l in sample_data]}")
    
    # Load models
    if args.model == 'all':
        model_keys = ['byol', 'dino', 'mocov2']
    else:
        model_keys = [args.model]
    
    models = {}
    for key in model_keys:
        print(f"Loading {key}...")
        models[key] = {
            'model': build_model(MODEL_CONFIGS[key]['checkpoint'], device),
            'display_name': MODEL_CONFIGS[key]['name']
        }
    
    # Create comparison figure
    output_path = os.path.join(args.output_dir, f'attribution_comparison_cifar100.png')
    create_comparison_figure(models, sample_data, device, norm_fn, output_path)
    
    print(f"\nDone! Figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
