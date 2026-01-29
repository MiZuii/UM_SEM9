import torch
import os
import cv2
import numpy as np
from PIL import Image
from utils.data_loader import get_dataloader
from utils.model_loader import get_model
from methods.gradcam import GradCAM
from methods.ig import IntegratedGradients
from methods.diet import DiET
from select_examples import load_examples_cache
import matplotlib.pyplot as plt

HIGHEST_STEP_NUM = 4
MASK_SCALE = 8

def save_result(heatmap_tensor, original_image_tensor, save_path):
    # Normalize heatmap
    hm = heatmap_tensor.squeeze().cpu().detach().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-10)
    hm = (hm * 255).astype(np.uint8)
    
    # Save Heatmap
    cv2.imwrite(save_path + "_heatmap.png", hm)
    Image.fromarray(hm).save(f"{save_path}_heatmap.png")
    
    # Save Blended
    cmap = plt.get_cmap('jet')
    hmc = cmap(hm)[:,:,:3]
    hmc = (hmc * 255).astype(np.uint8)
    Image.fromarray(hmc).save(f"{save_path}_colored.png")

    img_np = original_image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    heatmap_h, heatmap_w = hmc.shape[:2]
    original_image_resized = cv2.resize(img_np, (heatmap_w, heatmap_h))
    blended_image = cv2.addWeighted(original_image_resized, 0.6, hmc, 0.4, 0)
    Image.fromarray(blended_image).save(f"{save_path}_blended.png")

    # org image
    Image.fromarray(original_image_resized).save(f"{save_path}_original.png")

def run_single_experiment():
    MODELS = ['resnet18', 'resnet34']
    # MODELS = ['resnet18', 'resnet34', 'resnet50']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    METHODS = ['DiET', 'GradCAM', 'IG']
    EXAMPLE_TYPES = ['correct_confident', 'incorrect_confident', 'uncertain', 'arbitrary']
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ensure models are saved here: project_root/models/{model}_{dataset}_model.pth
    MODEL_DIR = "./models" 
    OUT_DIR = "./results/single"
    
    # Load cached examples
    examples_cache = load_examples_cache()

    for d_name in DATASETS:
        print(f"--- Dataset: {d_name} ---")
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=1)
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            
            # Construct path to weights
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)

            if model is None:
                print(f"Skipping \"{m_name}\" due to missing weights.")
                continue
            
            # Get examples for this model/dataset
            baseline_examples = examples_cache.get(d_name, {}).get(m_name, {}).get('baseline')
            if baseline_examples is None:
                print(f"No baseline examples found for {m_name} on {d_name}")
                continue
            
            for example_type in EXAMPLE_TYPES:
                target_idx = baseline_examples[example_type]
                if target_idx is None:
                    print(f"    No {example_type} example found")
                    continue
                
                # Get specific image
                img, label = test_loader.dataset[target_idx]
                d_img = img.unsqueeze(0).to(DEVICE)
                
                print(f"    [{example_type}] Example idx: {target_idx}")
                
                # Get model prediction
                model.eval()
                with torch.no_grad():
                    output = model(d_img)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1).item()
                    true_class_prob = pred_probs[0, label].item()
                
                print(f"      True class: {label}, Predicted class: {pred_class}, Probability for true class: {true_class_prob:.4f}")
                
                for method_name in METHODS:
                    print(f"      >> Method: {method_name}")
                    
                    if method_name == 'GradCAM':
                        explainer = GradCAM(model)
                        mask = explainer.explain(d_img)
                    elif method_name == 'IG':
                        explainer = IntegratedGradients(model)
                        mask = explainer.explain(d_img)
                    elif method_name == 'DiET':
                        explainer = DiET(model, device=DEVICE, mask_scale=MASK_SCALE)
                        mask = explainer.explain(d_img, label)
                    
                    # Save
                    save_dir = os.path.join(OUT_DIR, d_name, m_name, "baseline", example_type, method_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_result(mask, img, os.path.join(save_dir, f"idx_{target_idx}"))

            print(f"  > DiET version of model: {m_name}")

            # Construct path to weights
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_{HIGHEST_STEP_NUM}_diet_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)

            if model is None:
                print(f"Skipping diet version of \"{m_name}\" due to missing weights.")
                continue

            # Get examples for this model/dataset
            diet_examples = examples_cache.get(d_name, {}).get(m_name, {}).get('diet')
            if diet_examples is None:
                print(f"No DiET examples found for {m_name} on {d_name}")
                continue
            
            for example_type in EXAMPLE_TYPES:
                target_idx = diet_examples[example_type]
                if target_idx is None:
                    print(f"    No {example_type} example found")
                    continue
                
                # Get specific image
                img, label = test_loader.dataset[target_idx]
                d_img = img.unsqueeze(0).to(DEVICE)
                
                print(f"    [{example_type}] Example idx: {target_idx}")
                
                # Get model prediction
                model.eval()
                with torch.no_grad():
                    output = model(d_img)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1).item()
                    true_class_prob = pred_probs[0, label].item()
                
                print(f"      True class: {label}, Predicted class: {pred_class}, Probability for true class: {true_class_prob:.4f}")

                for method_name in METHODS:
                    print(f"      >> Method: {method_name}")
                    
                    if method_name == 'GradCAM':
                        explainer = GradCAM(model)
                        mask = explainer.explain(d_img)
                    elif method_name == 'IG':
                        explainer = IntegratedGradients(model)
                        mask = explainer.explain(d_img)
                    elif method_name == 'DiET':
                        explainer = DiET(model, device=DEVICE, mask_scale=MASK_SCALE)
                        mask = explainer.explain(d_img, label)
                    
                    # Save
                    save_dir = os.path.join(OUT_DIR, d_name, m_name, "diet", example_type, method_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_result(mask, img, os.path.join(save_dir, f"idx_{target_idx}"))


if __name__ == "__main__":
    run_single_experiment()