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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose

batch_sz = 16

# -------------------------------------------------------- #
#                 DATASET LOADING UTILITIES                #
# -------------------------------------------------------- #

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
#         DIET SINGULAR DISTILLATION IMPLEMENTATION        #
# -------------------------------------------------------- #

def print_mask_metrics(metrics):

    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "l1:", print_metrics[1], \
            "t2:", print_metrics[2], \
            "l0:", print_metrics[3], \
          flush=True)
    return

def update_mask(mask, x, model, mask_opt, simp_weight, device='cuda'):

    im_size = 224
    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=8, mode='bilinear')

    batch_D_d, y, pred_fb_d = x

    batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(device), ups(mask).to(device), pred_fb_d.to(device)

    # get random background color to replace masked pixels
    background_means = torch.ones((1, 3))*torch.Tensor([0.527, 0.447, 0.403])
    background_std = torch.ones((1, 3))*torch.Tensor([0.05, 0.05, 0.05])
    avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(device)
    # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

    pred_fs_d = sm(model(batch_D_d))
    pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

    # calculate loss by comparing the two models (t1) and the two datasets (t2)
    t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
    sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(im_size*im_size)
    loss = ((simp_weight*sim_heur + t2))

    mask_opt.zero_grad()
    loss.backward()
    mask_opt.step()

    with torch.no_grad():

        mask.copy_(mask.clamp(max=1, min=0))
        mask_l0_norm = (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu()

    metrics = torch.Tensor([loss.item(), torch.sum(sim_heur).item(), torch.sum(t2).item(), mask_l0_norm.item()])

    print_mask_metrics(metrics)
    return metrics

def distill_singular(mask, model, x, mask_opt):

    num_rounding_steps = 1
    # simp_weight = [1- r*(0.9/num_rounding_steps) for r in range(num_rounding_steps)]
    simp_weight = 1

    mask_converged = False
    prev_loss, prev_prev_loss = float('inf'), float('inf')

    while (not mask_converged):

        mask_metrics = update_mask(mask, x, model, mask_opt, simp_weight)
        mask_loss = mask_metrics[0]
        mask_converged = (mask_loss > 0.998*prev_prev_loss) and (mask_loss < 1.002*prev_prev_loss)
        
        prev_prev_loss = prev_loss
        prev_loss = mask_loss

    return mask


if __name__ == "__main__":
    idx = 1450
    im_size = 224

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

    diet_model_path = "DiET/mnist_ups8_outdir/fs_4.pth"
    diet_model = resnet34(weights=None)
    diet_model.fc = torch.nn.Linear(512, num_classes)

    diet_model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    diet_model = diet_model.to(device)
    diet_model.eval()

    model = base_model

    sm = torch.nn.Softmax(dim=1)
    x = test_loader.dataset[idx][1].unsqueeze(0).to(device)
    y = test_loader.dataset[idx][2]
    pred_f = sm(model(x))

    mask = torch.ones((1, 1, im_size//8, im_size//8))
    mask = mask.requires_grad_(True)
    mask_opt = torch.optim.SGD([mask], lr=10)
    mask_opt.zero_grad()

    mask = distill_singular(mask, model, (x, y, pred_f), mask_opt)

    # ---------------------------------------------------------------------------- #
    #                                    DRAWING                                   #
    # ---------------------------------------------------------------------------- #

    # generate heatmap from DiET mask
    ups = torch.nn.Upsample(scale_factor=8, mode='bilinear')
    heatmap_tensor = ups(mask)

    # convert tensor to a numpy array, removing extra dimensions
    heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()

    # normalize the heatmap to range [0, 1] for visualization
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)

    cmap = plt.get_cmap('jet') # 'jet' colormap goes from blue (low) to red (high)
    heatmap_colored = cmap(heatmap)[:,:,:3] # Get RGB, discard alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_colored_image = Image.fromarray(heatmap_colored)
    heatmap_colored_image.save("diet_mask_heatmap_colored.png")

    # --- Blend the colored heatmap with the original image ---
    # Retrieve the original image (datapoint at index 0) from the test_loader
    original_image_tensor = test_loader.dataset[idx][1] # C, H, W
    original_image_np = original_image_tensor.permute(1, 2, 0).cpu().numpy() # H, W, C
    original_image_np = (original_image_np * 255).astype(np.uint8) # Scale to 0-255

    # Resize the original image to match the heatmap size if necessary (should already be 224x224)
    # This step is good for robustness in case dimensions differ
    heatmap_h, heatmap_w = heatmap_colored.shape[:2]
    original_image_resized = cv2.resize(original_image_np, (heatmap_w, heatmap_h))

    # Blend the images using OpenCV's addWeighted
    # alpha is the weight for the first image, beta for the second
    # gamma is a scalar added to each sum
    blended_image = cv2.addWeighted(original_image_resized, 0.6, heatmap_colored, 0.4, 0)
    blended_image_pil = Image.fromarray(blended_image)
    blended_image_pil.save("diet_mask_heatmap_blended.png")

    original_image = Image.fromarray(original_image_resized)
    original_image.save("original_image.png")