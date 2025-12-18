import torch
import os
from tqdm import tqdm
from utils.data_loader import get_dataloader
from utils.model_loader import get_model
from methods.gradcam import GradCAM
from methods.ig import IntegratedGradients
from methods.diet import DiET

def run_batch_experiment():
    MODELS = ['resnet18', 'resnet34', 'resnet50']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    METHODS = ['DiET', 'GradCAM', 'IG']
    
    SUBSET_SIZE = 20 # Run on first 20 images for demonstration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = "./models" 
    OUT_DIR = "./results/batch"

    for d_name in DATASETS:
        print(f"--- Dataset: {d_name} ---")
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=1)
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)
            
            for method_name in METHODS:
                print(f"    >> Method: {method_name}")
                
                # Prepare directory for saving masks tensors
                save_dir = os.path.join(OUT_DIR, d_name, m_name, method_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # Instantiate Explainer
                if method_name == 'GradCAM': explainer = GradCAM(model)
                elif method_name == 'IG': explainer = IntegratedGradients(model)
                elif method_name == 'DiET': explainer = DiET(model)

                saved_masks = []
                
                # Iterate over subset
                for i, (idx, img, label) in enumerate(tqdm(test_loader, total=SUBSET_SIZE)):
                    if i >= SUBSET_SIZE: break
                    
                    img = img.to(DEVICE)
                    
                    # Generate explanation
                    if method_name == 'DiET':
                        mask = explainer.explain(img, steps=30, device=DEVICE) # Fewer steps for batch speed
                    else:
                        mask = explainer.explain(img)
                    
                    # Store cpu tensor to save memory
                    saved_masks.append(mask.cpu())

                # Save the stack of masks for analysis/metrics later
                torch.save(torch.cat(saved_masks), os.path.join(save_dir, "masks.pt"))
                print(f"    Saved masks to {save_dir}/masks.pt")

if __name__ == "__main__":
    run_batch_experiment()