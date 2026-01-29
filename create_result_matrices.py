import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = "./results/single"
MODELS = ['resnet18', 'resnet34']
# MODELS = ['resnet18', 'resnet34', 'resnet50']
DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
METHODS = ['DiET', 'GradCAM', 'IG']
EXAMPLE_TYPES = ['correct_confident', 'incorrect_confident', 'uncertain', 'arbitrary']
MATRIX_TYPES = ['baseline', 'diet']  # Two separate matrices
IMAGE_TYPE = 'blended'  # 'blended', 'heatmap', 'colored', or 'original'

def load_image_safe(img_path):
    """Load image safely, return None if not found"""
    if os.path.exists(img_path):
        try:
            return Image.open(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    return None

def create_matrix_for_dataset(dataset_name, matrix_type):
    """
    Create a matrix visualization for a dataset.
    X-axis: Methods (DiET, GradCAM, IG)
    Y-axis: Models (resnet18, resnet34, resnet50)
    """
    print(f"Creating matrix for {dataset_name} - {matrix_type}")
    
    # Collect images for each model-method combination
    images_grid = {}
    
    for model in MODELS:
        images_grid[model] = {}
        for method in METHODS:
            images_grid[model][method] = None
    
    # Load images for each model-method combo from "arbitrary" example type only
    for model in MODELS:
        for method in METHODS:
            img_path = os.path.join(
                RESULTS_DIR, dataset_name, model, matrix_type, 
                'arbitrary', method
            )
            
            if os.path.exists(img_path):
                # Find the image file
                for file in os.listdir(img_path):
                    if file.endswith(f"_{IMAGE_TYPE}.png"):
                        full_path = os.path.join(img_path, file)
                        img = load_image_safe(full_path)
                        if img is not None:
                            images_grid[model][method] = img
                            break
    
    # Create the matrix figure
    n_models = len(MODELS)
    n_methods = len(METHODS)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_models, n_methods, figsize=(14, 10))
    
    # Handle case where n_models or n_methods is 1 (axes is not 2D array)
    if n_models == 1: axes = np.array([axes])
    if n_methods == 1: axes = axes[:, np.newaxis] # Make it 2D
    if n_models == 1 and n_methods == 1: axes = np.array([[axes]])

    fig.suptitle(f"{dataset_name} - {matrix_type.upper()} ({IMAGE_TYPE})", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Fill in the matrix
    for i, model in enumerate(MODELS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            img = images_grid[model][method]
            
            # 1. Plot the image (or N/A placeholder)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            
            # 2. Turn off standard axis ticks/spines
            ax.axis('off')

            # 3. SET COLUMN HEADERS (Methods) - Only on the first row
            if i == 0:
                ax.set_title(method, fontsize=14, fontweight='bold', pad=15)

            # 4. SET ROW LABELS (Models) - Only on the first column
            if j == 0:
                ax.text(-0.1, 0.5, model, 
                        transform=ax.transAxes, 
                        fontsize=14, fontweight='bold',
                        va='center', ha='right', rotation=90)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]) 
    
    # Save the matrix
    output_dir = os.path.join(RESULTS_DIR, dataset_name, "matrices")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{matrix_type}_{IMAGE_TYPE}_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return output_path

def create_cross_dataset_matrix(matrix_type, method):
    """
    Create a matrix visualization comparing datasets vs models for a specific method.
    X-axis: Datasets (CIFAR10, CIFAR100, HardMNIST)
    Y-axis: Models (resnet18, resnet34)
    """
    print(f"Creating cross-dataset matrix for Method: {method} - Type: {matrix_type}")

    n_models = len(MODELS)
    n_datasets = len(DATASETS)

    # Create figure
    fig, axes = plt.subplots(n_models, n_datasets, figsize=(14, 10))

    # Handle axis array dimensions
    if n_models == 1: axes = np.array([axes])
    if n_datasets == 1: axes = axes[:, np.newaxis]
    if n_models == 1 and n_datasets == 1: axes = np.array([[axes]])

    fig.suptitle(f"Cross-Dataset {method} Analysis - {matrix_type.upper()} ({IMAGE_TYPE})", 
                 fontsize=20, fontweight='bold', y=0.98)

    # Fill in the matrix
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            
            # Construct path to specific image
            # Path: RESULTS_DIR / dataset / model / matrix_type / 'arbitrary' / method
            img_path = os.path.join(
                RESULTS_DIR, dataset, model, matrix_type, 
                'arbitrary', method
            )
            
            img = None
            if os.path.exists(img_path):
                for file in os.listdir(img_path):
                    if file.endswith(f"_{IMAGE_TYPE}.png"):
                        full_path = os.path.join(img_path, file)
                        img = load_image_safe(full_path)
                        if img: break
            
            # Plot
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            
            ax.axis('off')

            # Column Headers (Datasets)
            if i == 0:
                ax.set_title(dataset, fontsize=14, fontweight='bold', pad=15)
            
            # Row Labels (Models)
            if j == 0:
                ax.text(-0.1, 0.5, model, 
                        transform=ax.transAxes, 
                        fontsize=14, fontweight='bold',
                        va='center', ha='right', rotation=90)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # Save directly to results/single with method name in filename
    output_filename = f"{method}_Cross_Dataset_{matrix_type}_{IMAGE_TYPE}.png"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")

def create_dual_image_comparison(dataset_name):
    """
    Create comparison images showing baseline vs diet for each model/method.
    Saves two images side by side for each combination using "arbitrary" examples.
    """
    print(f"Creating dual comparisons for {dataset_name}")
    
    for model in MODELS:
        for method in METHODS:
            baseline_img = None
            diet_img = None
            
            # Find baseline image
            img_path = os.path.join(
                RESULTS_DIR, dataset_name, model, 'baseline', 
                'arbitrary', method
            )
            if os.path.exists(img_path):
                for file in os.listdir(img_path):
                    if file.endswith(f"_{IMAGE_TYPE}.png"):
                        baseline_img = load_image_safe(os.path.join(img_path, file))
                        if baseline_img: break
            
            # Find diet image
            img_path = os.path.join(
                RESULTS_DIR, dataset_name, model, 'diet', 
                'arbitrary', method
            )
            if os.path.exists(img_path):
                for file in os.listdir(img_path):
                    if file.endswith(f"_{IMAGE_TYPE}.png"):
                        diet_img = load_image_safe(os.path.join(img_path, file))
                        if diet_img: break
            
            if baseline_img and diet_img:
                total_width = baseline_img.width + diet_img.width + 20
                max_height = max(baseline_img.height, diet_img.height)
                
                comparison = Image.new('RGB', (total_width, max_height + 50), color='white')
                comparison.paste(baseline_img, (10, 50))
                comparison.paste(diet_img, (baseline_img.width + 10, 50))
                
                output_dir = os.path.join(RESULTS_DIR, dataset_name, 'comparisons')
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, f"{model}_{method}_comparison.png")
                comparison.save(output_path)
                print(f"  Saved: {output_path}")

def main():
    print("=" * 60)
    print("Creating Result Matrices from Experiment")
    print("=" * 60)
    
    # 1. Existing per-dataset matrices
    for dataset in DATASETS:
        dataset_path = os.path.join(RESULTS_DIR, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset directory not found: {dataset_path}")
            continue
        
        print(f"\nProcessing {dataset}:")
        print("-" * 40)
        
        for matrix_type in MATRIX_TYPES:
            create_matrix_for_dataset(dataset, matrix_type)
        
        create_dual_image_comparison(dataset)

    print("\n" + "-" * 40)
    print("Creating Cross-Dataset Matrices (All Methods)")
    print("-" * 40)

    # 2. New cross-dataset matrices for ALL methods
    for method in METHODS:
        print(f"\n--- Processing {method} ---")
        for matrix_type in MATRIX_TYPES:
            create_cross_dataset_matrix(matrix_type, method=method)
    
    print("\n" + "=" * 60)
    print("Matrix creation complete!")
    print(f"Results saved in: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()