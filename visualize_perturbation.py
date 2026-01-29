import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import get_dataloader
from utils.model_loader import get_model
from methods.gradcam import GradCAM
from methods.ig import IntegratedGradients
from methods.diet import DiET

# --- Configuration ---
RESULTS_DIR = "./results/perturbation"
CACHE_FILE = "./results/example_cache.json"
JSON_PATH = os.path.join(RESULTS_DIR, "perturbation_results.json")
MODEL_DIR = "./models"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters for visualization
MODELS = ['resnet18', 'resnet34']
DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
HIGHEST_STEP_NUM = 4
MASK_SCALE = 8
DISPLAY_RATIOS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

def load_example_cache():
    """Loads the dictionary of selected example indices."""
    if not os.path.exists(CACHE_FILE):
        print(f"Error: Cache file not found at {CACHE_FILE}")
        print("Please run 'select_examples.py' first.")
        return None
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

def get_perturbation_mask(img_tensor, mask, keep_ratio, device):
    """
    Applies the perturbation logic to a single image tensor for visualization.
    Returns the perturbed image as a numpy array (H, W, C).
    """
    if keep_ratio >= 1.0:
        return img_tensor.permute(1, 2, 0).cpu().numpy()

    # Move mask to correct device just in case
    mask = mask.to(device)

    # Flatten mask to find threshold
    flat_mask = mask.view(-1)
    total_pixels = flat_mask.numel()
    k = int(total_pixels * keep_ratio)
    
    # Find threshold
    if k <= 0:
        binary_mask = torch.zeros_like(mask)
    else:
        topk_vals, _ = torch.topk(flat_mask, k)
        threshold = topk_vals[-1]
        binary_mask = (mask >= threshold).float()
    
    # Average value (dataset mean)
    avg_val = torch.tensor([0.527, 0.447, 0.403], device=device).view(3, 1, 1)
    
    # Perturb
    perturbed_img = img_tensor * binary_mask + avg_val * (1 - binary_mask)
    
    # Convert to numpy for plotting
    img_np = perturbed_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    img_np = np.clip(img_np, 0, 1)
    return img_np

def generate_visualizations(example_cache):
    """
    Generates perturbation grids for ALL datasets and models using cached arbitrary indices.
    """
    print(f"--- Generating Visual Examples (Reading indices from {CACHE_FILE}) ---")
    
    for d_name in DATASETS:
        print(f"Processing Dataset: {d_name}")
        # Load dataset once per dataset loop
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=1)
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            
            # 1. Get the arbitrary index from cache
            try:
                # The cache structure is: cache[dataset][model]['baseline']['arbitrary']
                # (Assuming baseline and diet share the same test indices)
                target_idx = example_cache[d_name][m_name]['baseline']['arbitrary']
            except (KeyError, TypeError):
                print(f"    Skipping: No arbitrary index found in cache for {d_name}/{m_name}")
                continue

            print(f"    Using Arbitrary Index: {target_idx}")
            
            # 2. Get the specific image
            img, label = test_loader.dataset[target_idx]
            img_tensor = img.unsqueeze(0).to(DEVICE) # [1, C, H, W]

            # 3. Load Models
            baseline_path = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            diet_path = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_{HIGHEST_STEP_NUM}_diet_model.pth")
            
            model_baseline = get_model(m_name, num_classes, baseline_path, DEVICE)
            model_diet = get_model(m_name, num_classes, diet_path, DEVICE)
            
            if model_baseline is None or model_diet is None:
                print(f"    Skipping: One or both model weights missing.")
                continue

            # 4. Initialize Explainers
            # Baseline uses GradCAM and IG
            # DiET model uses DiET method
            explainers = {
                'GradCAM': (GradCAM(model_baseline), model_baseline),
                'IG':      (IntegratedGradients(model_baseline), model_baseline),
                'DiET':    (DiET(model_diet, device=DEVICE, mask_scale=MASK_SCALE), model_diet)
            }

            # 5. Create Grid Plot
            fig, axes = plt.subplots(len(explainers), len(DISPLAY_RATIOS), figsize=(16, 9))
            fig.suptitle(f"{d_name} - {m_name} (Idx: {target_idx})", fontsize=16, fontweight='bold')

            for i, (method_name, (explainer, model)) in enumerate(explainers.items()):
                model.eval()
                
                # Generate Mask
                if method_name == 'DiET':
                    mask = explainer.explain(img_tensor, label)
                else:
                    mask = explainer.explain(img_tensor)
                
                # Ensure mask is on device before processing
                mask = mask.to(DEVICE)

                for j, ratio in enumerate(DISPLAY_RATIOS):
                    ax = axes[i, j]
                    
                    # Create perturbed image
                    p_img = get_perturbation_mask(img_tensor.squeeze(), mask, ratio, DEVICE)
                    
                    ax.imshow(p_img)
                    ax.axis('off')
                    
                    # Row Labels (Methods)
                    if j == 0:
                        ax.text(-0.2, 0.5, method_name, transform=ax.transAxes, 
                                fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
                    
                    # Column Labels (Ratios)
                    if i == 0:
                        ax.set_title(f"Keep {int(ratio*100)}%", fontsize=12, fontweight='bold')

            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
            
            # Save file
            out_filename = f"vis_{d_name}_{m_name}_idx{target_idx}.png"
            out_path = os.path.join(RESULTS_DIR, out_filename)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved: {out_filename}")

def plot_accuracy_curves():
    """
    Reads the JSON result file and creates line plots for accuracy vs. sparsity.
    """
    print("\n--- Generating Accuracy Plots ---")
    
    if not os.path.exists(JSON_PATH):
        print(f"Warning: Results file not found at {JSON_PATH}")
        print("Please run 'experiment_perturbation.py' first to generate the data.")
        return

    with open(JSON_PATH, 'r') as f:
        results = json.load(f)

    # Plot for each dataset
    for d_name, d_data in results.items():
        # Figure setup: 1 Row, N Cols (where N is number of models found)
        model_names = list(d_data.keys())
        if not model_names:
            continue
            
        fig, axes = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 6))
        fig.suptitle(f"Perturbation Robustness - {d_name}", fontsize=18, fontweight='bold', y=0.98)
        
        # Ensure axes is iterable if there's only one model
        if len(model_names) == 1:
            axes = [axes]
        
        for idx, m_name in enumerate(model_names):
            ax = axes[idx]
            m_data = d_data[m_name]
            
            # --- Plot Baselines ---
            if 'baseline' in m_data:
                for method, scores in m_data['baseline'].items():
                    # Sort keys descending (1.0 -> 0.01)
                    ratios = sorted([float(k) for k in scores.keys()], reverse=True)
                    accs = [scores[str(r)] for r in ratios]
                    
                    # Style: Dashed lines for baselines
                    ax.plot(ratios, accs, marker='o', linestyle='--', linewidth=2, alpha=0.7, 
                            label=f"Baseline ({method})")

            # --- Plot DiET ---
            if 'diet' in m_data:
                for method, scores in m_data['diet'].items():
                    ratios = sorted([float(k) for k in scores.keys()], reverse=True)
                    accs = [scores[str(r)] for r in ratios]
                    
                    # Style: Solid bold line for DiET
                    ax.plot(ratios, accs, marker='s', linestyle='-', linewidth=3, 
                            label=f"DiET ({method})")

            # Axis formatting
            ax.set_title(m_name, fontsize=14, fontweight='bold')
            ax.set_xlabel("Fraction of Pixels Kept", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_xscale('log') 
            
            # Custom ticks for log scale matching our ratios
            ticks = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t*100)}%" for t in ticks])
            
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.invert_xaxis() # Standard convention: 100% (left) -> 1% (right)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = os.path.join(RESULTS_DIR, f"accuracy_plot_{d_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Saved plot to: {out_path}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load cache
    cache = load_example_cache()
    
    if cache:
        # 1. Create the visual grids for ALL datasets/models
        generate_visualizations(cache)
    
    # 2. Create the plots (Reads JSON)
    plot_accuracy_curves()