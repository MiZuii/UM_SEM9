"""
Download CIFAR100 and organize it in ImageFolder format for DINO training.

DINO expects data in ImageFolder format:
    data_path/
        class_0/
            img1.png
            img2.png
            ...
        class_1/
            ...

CIFAR100 has 100 fine-grained classes grouped into 20 coarse categories.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets


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


def download_and_convert_cifar100(output_dir: str, include_test: bool = False):
    """Download CIFAR100 and convert to ImageFolder format."""
    output_path = Path(output_dir)
    train_path = output_path / 'train'
    
    # Create class directories for train
    for class_name in CIFAR100_CLASSES:
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Download CIFAR100 train set (cache stored in parent to avoid ImageFolder issues)
    cache_dir = output_path.parent / '.cifar100_cache'
    print("Downloading CIFAR100 training set...")
    train_dataset = datasets.CIFAR100(
        root=str(cache_dir),
        train=True,
        download=True
    )
    
    # Save training images
    print(f"Saving {len(train_dataset)} training images...")
    for idx, (img, label) in enumerate(train_dataset):
        class_name = CIFAR100_CLASSES[label]
        img_path = train_path / class_name / f'{idx:05d}.png'
        img.save(img_path)
        
        if (idx + 1) % 10000 == 0:
            print(f"  Saved {idx + 1}/{len(train_dataset)} images")
    
    print(f"Training images saved to: {train_path}")
    
    if include_test:
        test_path = output_path / 'test'
        
        # Create class directories for test
        for class_name in CIFAR100_CLASSES:
            (test_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # Download CIFAR100 test set
        print("Downloading CIFAR100 test set...")
        test_dataset = datasets.CIFAR100(
            root=str(cache_dir),
            train=False,
            download=True
        )
        
        # Save test images
        print(f"Saving {len(test_dataset)} test images...")
        for idx, (img, label) in enumerate(test_dataset):
            class_name = CIFAR100_CLASSES[label]
            img_path = test_path / class_name / f'{idx:05d}.png'
            img.save(img_path)
        
        print(f"Test images saved to: {test_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("CIFAR100 ImageFolder Dataset Created")
    print("=" * 50)
    print(f"Location: {output_path}")
    print(f"Training images: {len(train_dataset)}")
    if include_test:
        print(f"Test images: {len(test_dataset)}")
    print(f"Classes: {len(CIFAR100_CLASSES)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Download CIFAR100 and convert to ImageFolder format'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/cifar100_imagefolder',
        help='Output directory for ImageFolder dataset'
    )
    parser.add_argument(
        '--include-test',
        action='store_true',
        help='Also convert test set (not needed for SSL pretraining)'
    )
    
    args = parser.parse_args()
    
    download_and_convert_cifar100(args.output, args.include_test)


if __name__ == '__main__':
    main()
