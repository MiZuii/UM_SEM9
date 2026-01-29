#!/usr/bin/env python3
"""
Create GridPG images from confident CIFAR10 images.
Creates 2x2 and 3x3 grids of different class images for GridPG evaluation.

Usage:
    python create_point_grid_dataset_cifar10.py --input_dir confident_images_cifar10_byol
"""
import os
import argparse
import random
from PIL import Image
import numpy as np
from glob import glob


def create_grid_image(images, grid_size):
    """Create a grid image from a list of PIL images."""
    img_size = images[0].size[0]  # Assuming square images
    grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))
    
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * img_size, row * img_size))
    
    return grid_img


def main():
    parser = argparse.ArgumentParser(description='Create GridPG images for CIFAR10')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing confident images')
    parser.add_argument('--output_dir_2x2', type=str, default=None,
                        help='Output directory for 2x2 grids')
    parser.add_argument('--output_dir_3x3', type=str, default=None,
                        help='Output directory for 3x3 grids')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of grid images to create (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directories
    if args.output_dir_2x2 is None:
        args.output_dir_2x2 = args.input_dir.replace('confident_images', 'grid_pg_images_2x2')
    if args.output_dir_3x3 is None:
        args.output_dir_3x3 = args.input_dir.replace('confident_images', 'grid_pg_images_3x3')
    
    os.makedirs(args.output_dir_2x2, exist_ok=True)
    os.makedirs(args.output_dir_3x3, exist_ok=True)

    # Load images grouped by class
    class_images = {i: [] for i in range(10)}
    for img_path in glob(os.path.join(args.input_dir, '*.png')):
        fname = os.path.basename(img_path)
        # Format: {sample_index}_{class_id}.png
        class_id = int(fname.split('_')[1].replace('.png', ''))
        class_images[class_id].append(img_path)

    print(f"Loaded images per class:")
    for i in range(10):
        print(f"  Class {i}: {len(class_images[i])} images")

    # Create 2x2 grids (4 images, 4 classes)
    print(f"\nCreating {args.num_samples} 2x2 grid images...")
    for sample_idx in range(args.num_samples):
        # Select 4 random classes
        selected_classes = random.sample(range(10), 4)
        images = []
        labels = []
        
        for cls in selected_classes:
            if len(class_images[cls]) == 0:
                continue
            img_path = random.choice(class_images[cls])
            images.append(Image.open(img_path))
            labels.append(cls)
        
        if len(images) == 4:
            grid_img = create_grid_image(images, 2)
            # Filename: {sample_idx}_{label0}_{label1}_{label2}_{label3}.png
            labels_str = '_'.join(map(str, labels))
            fname = f"{sample_idx}_{labels_str}.png"
            grid_img.save(os.path.join(args.output_dir_2x2, fname))

    # Create file list for 2x2
    list_file_2x2 = args.output_dir_2x2 + '_list.txt'
    with open(list_file_2x2, 'w') as f:
        for fname in sorted(os.listdir(args.output_dir_2x2)):
            if fname.endswith('.png'):
                f.write(fname + '\n')

    print(f"  Saved 2x2 grids to {args.output_dir_2x2}")

    # Create 3x3 grids (9 images, 9 classes)
    print(f"\nCreating {args.num_samples} 3x3 grid images...")
    for sample_idx in range(args.num_samples):
        # Select 9 random classes (with replacement since we have 10 classes)
        selected_classes = random.choices(range(10), k=9)
        images = []
        labels = []
        
        for cls in selected_classes:
            if len(class_images[cls]) == 0:
                continue
            img_path = random.choice(class_images[cls])
            images.append(Image.open(img_path))
            labels.append(cls)
        
        if len(images) == 9:
            grid_img = create_grid_image(images, 3)
            # Filename: {sample_idx}_{labels}.png
            labels_str = '_'.join(map(str, labels))
            fname = f"{sample_idx}_{labels_str}.png"
            grid_img.save(os.path.join(args.output_dir_3x3, fname))

    # Create file list for 3x3
    list_file_3x3 = args.output_dir_3x3 + '_list.txt'
    with open(list_file_3x3, 'w') as f:
        for fname in sorted(os.listdir(args.output_dir_3x3)):
            if fname.endswith('.png'):
                f.write(fname + '\n')

    print(f"  Saved 3x3 grids to {args.output_dir_3x3}")
    print("\nDone!")


if __name__ == '__main__':
    main()
