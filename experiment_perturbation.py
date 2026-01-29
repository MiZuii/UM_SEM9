import torch
import os
import numpy as np
import json
from tqdm import tqdm
from utils.data_loader import get_dataloader
from utils.model_loader import get_model
from methods.gradcam import GradCAM
from methods.ig import IntegratedGradients
from methods.diet import DiET

# Configuration matching experiment_single.py
HIGHEST_STEP_NUM = 4
MASK_SCALE = 8

# Perturbation parameters
KEEP_RATIOS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0] 
MAX_SAMPLES = 100 

def perturb_batch(images, masks, keep_ratio, device):
    """
    Adapts the logic from simplify_dataset in pixel_perturbation.py
    Keeps top (keep_ratio) percent of pixels based on mask importance.
    Fills the rest with an average gray value.
    """
    if keep_ratio >= 1.0:
        return images
    
    # --- FIX: Ensure masks are on the correct device ---
    masks = masks.to(device) 
    # ---------------------------------------------------

    batch_size, channels, height, width = images.shape
    total_pixels = height * width
    
    # Calculate how many pixels to keep
    k = int(total_pixels * keep_ratio)
    
    # Flatten mask to find thresholds
    # mask shape: [B, 1, H, W] -> [B, H*W]
    flat_masks = masks.view(batch_size, -1)
    
    # Find the k-th largest value in each mask
    topk_vals, _ = torch.topk(flat_masks, k, dim=1)
    thresholds = topk_vals[:, -1].view(batch_size, 1, 1, 1) # Smallest value in the top k
    
    # Create binary mask (1 for keep, 0 for remove)
    binary_mask = (masks >= thresholds).float()
    
    # Average value used in your pixel_perturbation.py
    avg_val = torch.tensor([0.527, 0.447, 0.403], device=device).view(1, 3, 1, 1)
    
    # Perturb
    perturbed_images = images * binary_mask + avg_val * (1 - binary_mask)
    
    return perturbed_images

def evaluate_method(model, loader, method_name, device, num_classes):
    """
    Runs perturbation test for a specific model and explanation method.
    """
    model.eval()
    
    # Initialize explainer
    explainer = None
    if method_name == 'GradCAM':
        explainer = GradCAM(model)
    elif method_name == 'IG':
        explainer = IntegratedGradients(model)
    elif method_name == 'DiET':
        explainer = DiET(model, device=device, mask_scale=MASK_SCALE)
    
    results = {r: {'correct': 0, 'total': 0} for r in KEEP_RATIOS}
    
    processed_count = 0
    
    print(f"      >> Running evaluation for {method_name}...")
    
    # Iterate through dataset
    for img_batch, label_batch in tqdm(loader, leave=False):
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        
        # 1. Generate Masks
        masks_list = []
        for i in range(img_batch.shape[0]):
            img_single = img_batch[i].unsqueeze(0)
            target = label_batch[i].item()
            
            if method_name == 'DiET':
                m = explainer.explain(img_single, target)
            else:
                m = explainer.explain(img_single)
            
            masks_list.append(m)
            
        masks = torch.cat(masks_list, dim=0) # [B, 1, H, W]

        # 2. Evaluate at different perturbation ratios
        for ratio in KEEP_RATIOS:
            perturbed_imgs = perturb_batch(img_batch, masks, ratio, device)
            
            with torch.no_grad():
                output = model(perturbed_imgs)
                preds = torch.argmax(output, dim=1)
                
            correct = (preds == label_batch).sum().item()
            results[ratio]['correct'] += correct
            results[ratio]['total'] += len(label_batch)

        processed_count += len(label_batch)
        if MAX_SAMPLES and processed_count >= MAX_SAMPLES:
            break

    # Calculate final accuracies
    final_acc = {}
    for r in results:
        if results[r]['total'] > 0:
            final_acc[r] = results[r]['correct'] / results[r]['total']
        else:
            final_acc[r] = 0.0
            
    return final_acc

def run_perturbation_experiment():
    MODELS = ['resnet18', 'resnet34']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    METHODS_STANDARD = ['GradCAM', 'IG'] 
    METHODS_DIET = ['DiET']
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MODEL_DIR = "./models"
    OUT_DIR = "./results/perturbation"
    os.makedirs(OUT_DIR, exist_ok=True)

    experiment_results = {}

    for d_name in DATASETS:
        print(f"--- Dataset: {d_name} ---")
        experiment_results[d_name] = {}
        
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=32)
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            experiment_results[d_name][m_name] = {'baseline': {}, 'diet': {}}
            
            # 1. Standard Models (Baseline)
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)

            if model is None:
                print(f"Skipping \"{m_name}\" (Baseline) due to missing weights.")
            else:
                for method in METHODS_STANDARD:
                    print(f"    Method: {method}")
                    acc_curve = evaluate_method(model, test_loader, method, DEVICE, num_classes)
                    experiment_results[d_name][m_name]['baseline'][method] = acc_curve
                    print(f"      Result: {acc_curve}")

            # 2. DiET Models
            print(f"  > DiET version of model: {m_name}")
            weights_diet = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_{HIGHEST_STEP_NUM}_diet_model.pth")
            model_diet = get_model(m_name, num_classes, weights_diet, DEVICE)

            if model_diet is None:
                print(f"Skipping \"{m_name}\" (DiET) due to missing weights.")
            else:
                for method in METHODS_DIET:
                    print(f"    Method: {method}")
                    acc_curve = evaluate_method(model_diet, test_loader, method, DEVICE, num_classes)
                    experiment_results[d_name][m_name]['diet'][method] = acc_curve
                    print(f"      Result: {acc_curve}")
            
            # Save results progressively
            with open(os.path.join(OUT_DIR, "perturbation_results.json"), 'w') as f:
                json.dump(experiment_results, f, indent=4)

    print(f"Experiment finished. Results saved to {os.path.join(OUT_DIR, 'perturbation_results.json')}")

if __name__ == "__main__":
    run_perturbation_experiment()