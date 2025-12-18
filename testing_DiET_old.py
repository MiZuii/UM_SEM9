from torchvision.models import resnet34
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose


# -------------------------------------------------------- #
#                 DATASET LOADING UTILITIES                #
# -------------------------------------------------------- #

batch_sz = 16

class DatasetfromDisk(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        target_resolution = (224, 224)
        self.transform = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]).convert('RGB'))
        
        return idx, image, self.labels[idx]

def load_mnist_from_disk(data_path):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "training/*/*")
    test_files = glob.glob(data_path + "testing/*/*")

    for f in train_files:
        if f[-3:] != "png":
            continue
        
        train_imgs.append(f)
        train_labels.append(int(f.split("/")[-2]))

    for f in test_files:
        if f[-3:] != "png":
            continue
        
        test_imgs.append(f)
        test_labels.append(int(f.split("/")[-2]))


    return train_imgs, train_labels, test_imgs, test_labels

# -------------------------------------------------------- #
#                  GRADCAM IMPLEMENTATION                  #
# -------------------------------------------------------- #

class GradCAMExtractor:
    #Extract tensors needed for Gradcam using hooks
    
    def __init__(self, model):
        self.model = model

        self.features = None
        self.feat_grad = None

        prev_module = None
        self.target_module = None

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                prev_module = m
            elif isinstance(m, nn.Linear):
                self.target_module = prev_module
                break

        if self.target_module is not None:
            # Register feature-gradient and feature hooks for each layer
            handle_g = self.target_module.register_backward_hook(self._extract_layer_grads)
            handle_f = self.target_module.register_forward_hook(self._extract_layer_features)

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        self.feature_grads = out_grad[0]
    
    def _extract_layer_features(self, module, input, output):
        # function to collect the layer outputs
        self.features = output

    def getFeaturesAndGrads(self, x, target_class):

        out = self.model(x)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward()

        return self.features, self.feature_grads


class GradCAM():
    """
    Compute GradCAM 
    """

    def __init__(self, model):
        self.model = model
        self.model_ext = GradCAMExtractor(self.model)


    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        features, intermed_grad = self.model_ext.getFeaturesAndGrads(image, target_class=target_class)

        # GradCAM computation
        grads = intermed_grad.mean(dim=(2,3), keepdim=True)
        cam = (F.relu(features)* grads).sum(1, keepdim=True)
        cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)
        return cam_resized
    
# -------------------------------------------------------- #
#            INTEGRATED GRADIENTS IMPLEMENTATION           #
# -------------------------------------------------------- #

class IntegratedGradients():

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate scaled inputs: shape (steps, C, H, W)
        # Baseline is assumed to be black (zeros)
        step_list = torch.arange(steps + 1, device=input_image.device).view(-1, 1, 1, 1) / steps
        return input_image * step_list

    def saliency(self, input_image, target_class=None, steps=50):
        # Determine target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_image)
                target_class = output.argmax(dim=1).item()

        # Generate interpolated images
        scaled_images = self.generate_images_on_linear_path(input_image, steps)
        scaled_images.requires_grad = True

        # Run forward pass on all scaled images (batch processing)
        # Note: If memory is an issue, this can be chunked
        output = self.model(scaled_images)
        
        # Get score for the target class
        score = output[:, target_class].sum()
        
        # Clear previous grads
        self.model.zero_grad()
        
        # Backward pass to get gradients
        score.backward()
        
        # gradients shape: (steps+1, C, H, W)
        gradients = scaled_images.grad
        
        # Average gradients (Riemann approximation of integral)
        avg_gradients = torch.mean(gradients[:-1], dim=0)

        # IG = (Input - Baseline) * AvgGrad
        # Since Baseline is 0, IG = Input * AvgGrad
        integrated_gradients = input_image.squeeze(0) * avg_gradients

        # Convert to single channel saliency map (average across RGB channels)
        saliency_map = torch.mean(torch.abs(integrated_gradients), dim=0).unsqueeze(0).unsqueeze(0)
        
        return saliency_map
    
# -------------------------------------------------------- #
#                          HELPERS                         #
# -------------------------------------------------------- #

def save_heatmap_and_blend(heatmap_tensor, original_image_tensor, output_prefix, blend=True):
    """
    Saves a raw heatmap and optionally a blended version (like DiET presentation).
    """
    # 1. Prepare Heatmap
    heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    
    # Save raw grayscale heatmap (Like GradCAM)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(f"{output_prefix}_grayscale.png")

    if blend:
        # 2. Prepare Colored Heatmap (Like DiET)
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)[:,:,:3] 
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        Image.fromarray(heatmap_colored).save(f"{output_prefix}_colored.png")

        # 3. Blend with Original
        original_image_np = original_image_tensor.permute(1, 2, 0).cpu().numpy()
        original_image_np = (original_image_np * 255).astype(np.uint8)
        
        heatmap_h, heatmap_w = heatmap_colored.shape[:2]
        original_image_resized = cv2.resize(original_image_np, (heatmap_w, heatmap_h))
        
        blended_image = cv2.addWeighted(original_image_resized, 0.6, heatmap_colored, 0.4, 0)
        Image.fromarray(blended_image).save(f"{output_prefix}_blended.png")

# --- Testing Functions ---

def test_model_gradcam(model, test_loader, idx, device, model_name, out="test"):

    # create dir gradcam
    os.makedirs(f"{out}/gradcam", exist_ok=True)

    tested_image = test_loader.dataset[idx][1].unsqueeze(0).to(device)
    tested_label = test_loader.dataset[idx][2]

    GradCAM_computer = GradCAM(model)
    cam = GradCAM_computer.saliency(tested_image, target_class=torch.tensor([tested_label]).to(device))
    
    # Using the standardized visualizer now
    original_img = test_loader.dataset[idx][1]
    save_heatmap_and_blend(cam, original_img, f"{out}/gradcam/{model_name}", blend=True)

def test_model_ig(model, test_loader, idx, device, model_name, out="test"):
    os.makedirs(f"{out}/ig", exist_ok=True)
    tested_image = test_loader.dataset[idx][1].unsqueeze(0).to(device)
    tested_label = test_loader.dataset[idx][2]

    IG_computer = IntegratedGradients(model)
    # steps=50 is usually sufficient for ResNet
    ig_map = IG_computer.saliency(tested_image, target_class=tested_label, steps=50)

    # Visualize similar to DiET (Colored + Blended)
    original_img = test_loader.dataset[idx][1]
    save_heatmap_and_blend(ig_map, original_img, f"{out}/ig/{model_name}", blend=True)


if __name__ == "__main__":

    idx = 1450

    device = "cuda"

    # load full dataset
    data_dir = "DiET/data/hard_mnist/"
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_from_disk(data_dir)
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
    test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)


    # load model
    base_model_path = "DiET/trained_models/hard_mnist_rn34.pth"
    base_model = resnet34(weights=None)
    base_model.fc = torch.nn.Linear(512, num_classes)

    base_model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    base_model = base_model.to(device)
    base_model.eval()

    cmp_model_path = "DiET/mnist_ups8_outdir/fs_1.pth"
    cmp_model_path = resnet34(weights=None)
    cmp_model_path.fc = torch.nn.Linear(512, num_classes)

    cmp_model_path.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    cmp_model_path = cmp_model_path.to(device)
    cmp_model_path.eval()

    print("Testing gradcam")
    test_model_gradcam(base_model, test_loader, idx, device, "base_model")
    test_model_gradcam(cmp_model_path, test_loader, idx, device, "diet_model")

    print("Testing IG")
    test_model_ig(base_model, test_loader, idx, device, "base_model")
    test_model_ig(cmp_model_path, test_loader, idx, device, "diet_model")

    print("Generating heatmap for DiET")
    # Get mask for the idx from calculated masks
    ups = torch.nn.Upsample(scale_factor=8, mode='bilinear')
    test_mask = torch.load("DiET/mnist_ups8_outdir/test_mask.pt")

    mask_for_image = test_mask[idx]

    # upsample the mask to the image resolution (224x224)
    # .unsqueeze(0).unsqueeze(0) adds batch and channel dimensions required by nn.Upsample
    heatmap_tensor = ups(mask_for_image.unsqueeze(0))
    original_img = test_loader.dataset[idx][1]
    save_heatmap_and_blend(heatmap_tensor, original_img, "test/diet_mask_heatmap", blend=True)


    original_img = test_loader.dataset[idx][1]
    original_img_np = original_img.permute(1, 2, 0).cpu().numpy()
    original_img_np = (original_img_np * 255).astype(np.uint8)
    Image.fromarray(original_img_np).save("test/original_image.png")