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
import matplotlib.pyplot as plt

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
    MODELS = ['resnet18', 'resnet34', 'resnet50', 'diet']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    METHODS = ['DiET', 'GradCAM', 'IG']
    
    TARGET_IDX = 1450 # Arbitrary index to test
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ensure models are saved here: project_root/models/{model}_{dataset}_model.pth
    MODEL_DIR = "./models" 
    OUT_DIR = "./results/single"

    for d_name in DATASETS:
        print(f"--- Dataset: {d_name} ---")
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=1)
        
        # Get specific image
        img, label = test_loader.dataset[TARGET_IDX]
        d_img = img.unsqueeze(0).to(DEVICE)
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            
            # Construct path to weights
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)

            if model is None:
                print(f"Skipping \"{m_name}\" due to missing weights.")
                continue
            
            for method_name in METHODS:
                print(f"    >> Method: {method_name}")
                
                if method_name == 'GradCAM':
                    explainer = GradCAM(model)
                    mask = explainer.explain(d_img)
                elif method_name == 'IG':
                    explainer = IntegratedGradients(model)
                    mask = explainer.explain(d_img)
                elif method_name == 'DiET':
                    explainer = DiET(model, device=DEVICE)
                    mask = explainer.explain(d_img, label)
                
                # Save
                save_dir = os.path.join(OUT_DIR, d_name, m_name, method_name)
                os.makedirs(save_dir, exist_ok=True)
                save_result(mask, img, os.path.join(save_dir, f"idx_{TARGET_IDX}"))

if __name__ == "__main__":
    run_single_experiment()