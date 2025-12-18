import torch
import torch.nn as nn
import torchvision.models as models
import os

custom_model_to_base_map = {
    'diet': "resnet34"
}

def get_model(model_name, num_classes, weights_path=None, device='cuda'):
    """
    Load architecture and optionally load saved weights.
    """
    if model_name in ['resnet18', 'resnet34', 'resnet50']:
        model = getattr(models, model_name)(weights=None)
    elif model_name in ['diet']:
        model = getattr(models, custom_model_to_base_map[model_name])(weights=None)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Initialize model with no pre-trained weights initially to match training script logic
    # or use weights="DEFAULT" if you want ImageNet initialization
    
    # Modify FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load trained weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print(f"Warning: No weights found at {weights_path}, using random init.")
        return None

    model = model.to(device)
    model.eval()
    return model