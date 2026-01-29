import torch
import os
import json
import numpy as np
from utils.data_loader import get_dataloader
from utils.model_loader import get_model

HIGHEST_STEP_NUM = 4
CACHE_FILE = "./results/example_cache.json"
ARBITRARY_IDX = 1450  # Constant arbitrary index for all models/datasets

def select_examples_for_model(model, test_loader, num_classes, device):
    """
    Select 3 example indices:
    1. Correct prediction with high confidence
    2. Incorrect prediction with high confidence (in wrong class)
    3. Uncertain prediction (low confidence)
    
    Returns a dict with keys: 'correct_confident', 'incorrect_confident', 'uncertain'
    """
    model.eval()
    
    correct_confident = None
    incorrect_confident = None
    uncertain = None
    
    correct_confident_score = -1
    incorrect_confident_score = -1
    uncertain_score = 1.0  # Start high, we want the lowest uncertainty
    
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            if idx >= len(test_loader.dataset):
                break
                
            img = img.to(device)
            output = model(img)
            pred_probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(pred_probs, dim=1).item()
            max_prob = pred_probs[0, pred_class].item()
            true_class_prob = pred_probs[0, label].item()
            
            # Case 1: Correct prediction with high confidence
            if pred_class == label:
                if max_prob > correct_confident_score:
                    correct_confident_score = max_prob
                    correct_confident = idx
            
            # Case 2: Incorrect prediction with high confidence
            elif pred_class != label:
                if max_prob > incorrect_confident_score:
                    incorrect_confident_score = max_prob
                    incorrect_confident = idx
            
                # Case 3: Uncertain (closest to 0.5 max probability or lowest confidence difference)
                uncertainty = abs(max_prob - true_class_prob)
                if uncertainty < uncertain_score:
                    uncertain_score = uncertainty
                    uncertain = idx
    
    return {
        'correct_confident': correct_confident,
        'incorrect_confident': incorrect_confident,
        'uncertain': uncertain,
        'arbitrary': ARBITRARY_IDX
    }

def cache_examples():
    """
    Build and cache example selections for all model/dataset combinations.
    """
    MODELS = ['resnet18', 'resnet34']
    # MODELS = ['resnet18', 'resnet34', 'resnet50']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = "./models"
    
    examples_cache = {}
    
    for d_name in DATASETS:
        print(f"Processing Dataset: {d_name}")
        _, test_loader, num_classes = get_dataloader(d_name, batch_size=1)
        
        examples_cache[d_name] = {}
        
        for m_name in MODELS:
            print(f"  > Model: {m_name}")
            
            examples_cache[d_name][m_name] = {}
            
            # Baseline model
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)
            
            if model is None:
                print(f"    Skipping baseline \"{m_name}\" due to missing weights.")
                examples_cache[d_name][m_name]['baseline'] = None
            else:
                print(f"    Selecting examples for baseline {m_name}")
                examples = select_examples_for_model(model, test_loader, num_classes, DEVICE)
                examples_cache[d_name][m_name]['baseline'] = examples
                print(f"      - Correct confident: {examples['correct_confident']}")
                print(f"      - Incorrect confident: {examples['incorrect_confident']}")
                print(f"      - Uncertain: {examples['uncertain']}")
            
            # DiET model
            weights = os.path.join(MODEL_DIR, f"{m_name}_{d_name}_{HIGHEST_STEP_NUM}_diet_model.pth")
            model = get_model(m_name, num_classes, weights, DEVICE)
            
            if model is None:
                print(f"    Skipping DiET \"{m_name}\" due to missing weights.")
                examples_cache[d_name][m_name]['diet'] = None
            else:
                print(f"    Selecting examples for DiET {m_name}")
                examples = select_examples_for_model(model, test_loader, num_classes, DEVICE)
                examples_cache[d_name][m_name]['diet'] = examples
                print(f"      - Correct confident: {examples['correct_confident']}")
                print(f"      - Incorrect confident: {examples['incorrect_confident']}")
                print(f"      - Uncertain: {examples['uncertain']}")
    
    # Save cache
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(examples_cache, f, indent=2)
    
    print(f"\nCache saved to {CACHE_FILE}")
    return examples_cache

def load_examples_cache():
    """
    Load the cached examples from file.
    """
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Cache file not found: {CACHE_FILE}. Run 'python select_examples.py' first.")
    
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    cache_examples()
