import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
import glob
import time
import copy

# -------------------------------------------------------- #
#                  DATASET CLASSES                         #
# -------------------------------------------------------- #

class HardMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True, transform=None):
        self.files = []
        self.labels = []
        subset = "training" if is_train else "testing"
        
        # Search assumes structure: data_path/training/class_idx/img.png
        file_list = glob.glob(os.path.join(data_path, subset, "*", "*.png"))
        
        for f in file_list:
            self.files.append(f)
            # Assuming label is the parent folder name
            self.labels.append(int(f.split(os.sep)[-2]))
            
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class DiETDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper to make standard datasets compatible with DiET training loops.
    Returns: idx, image, label, prediction
    """
    def __init__(self, original_dataset, predictions):
        self.dataset = original_dataset
        self.predictions = predictions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Standard dataset returns (img, label)
        img, label = self.dataset[idx]
        pred = self.predictions[idx]
        return idx, img, label, pred

# -------------------------------------------------------- #
#               DiET CORE LOGIC (UNTOUCHED)                #
# -------------------------------------------------------- #

def print_mask_metrics(metrics):
    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "l1:", print_metrics[1], \
            "t1:", print_metrics[2], \
            "t2:", print_metrics[3], \
            "t1_acc:", print_metrics[4], \
            "fs_s_acc:", print_metrics[5], \
            "l0:", print_metrics[6], \
          flush=True)
    return

def print_model_metrics(metrics):
    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "t1:", print_metrics[1], \
            "t2:", print_metrics[2], \
            "t1_acc:", print_metrics[3], \
            "fs_s_acc:", print_metrics[4], \
            flush=True)
    return

def update_mask(mask, data_loader, model, mask_opt, simp_weight, args):

    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    metrics = torch.zeros((7,))
    num_samples = 0

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

        # get random background color to replace masked pixels
        background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
        background_std = torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
        avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
        # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(args.im_size*args.im_size)
        loss = ((simp_weight*sim_heur + t1 + t2)/len(batch_D_d))

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

        with torch.no_grad():

            mask.copy_(mask.clamp(max=1, min=0))

            t1_acc = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc = torch.sum(t1_acc).detach().cpu()
            fs_s_acc = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc = torch.sum(fs_s_acc).detach().cpu()
            mask_l0_norm = (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu()

        metrics += torch.Tensor([loss.item()*len(batch_D_d), torch.sum(sim_heur).item(), torch.sum(t1).item(), torch.sum(t2).item(), t1_acc.item(), fs_s_acc.item(), mask_l0_norm.item()])
        num_samples += len(batch_labels)

    metrics /= num_samples
    print_mask_metrics(metrics)
    return metrics

def update_mask_light(mask, data_loader, model, mask_opt, simp_weight, args):

    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')

    total_loss = 0.0
    num_samples = 0

    bg_mean = torch.tensor([0.527, 0.447, 0.403], device=args.device).view(1, 3, 1, 1)
    bg_std  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d = batch_D_d.to(args.device)
        pred_fb_d = pred_fb_d.to(args.device)
        batch_mask = ups(mask[idx]).to(args.device)

        # get random background color to replace masked pixels
        current_batch_size = len(idx)
        noise = torch.normal(mean=bg_mean.expand(current_batch_size, -1, -1, -1), 
                             std=bg_std.expand(current_batch_size, -1, -1, -1))
        avg_val = noise.clamp(max=1, min=0)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(args.im_size*args.im_size)
        loss = ((simp_weight*sim_heur + t1 + t2)/len(batch_D_d))

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

        with torch.no_grad():
            mask.copy_(mask.clamp(max=1, min=0))

        total_loss += loss.item() * current_batch_size
        num_samples += current_batch_size

    avg_loss = total_loss / num_samples
    return [avg_loss]

def update_model(mask, data_loader, model, model_opt, args):

    mask = mask.requires_grad_(False)
    model.train()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    
    num_samples = 0
    metrics = torch.zeros((5,))

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

        # get random background color to replace masked pixels
        background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
        background_std = torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
        avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
        # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        loss = ((t1 + t2)/len(batch_D_d))

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()


        with torch.no_grad():
            t1_acc = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc = torch.sum(t1_acc).detach().cpu()
            fs_s_acc = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc = torch.sum(fs_s_acc).detach().cpu()
            
        metrics += torch.Tensor([loss.item()*len(batch_D_d), torch.sum(t1).item(), torch.sum(t2).item(), t1_acc.item(), fs_s_acc.item()])
        num_samples += len(batch_labels)

    metrics /= num_samples
    print_model_metrics(metrics)

    return metrics

def update_model_light(mask, data_loader, model, model_opt, args):

    mask = mask.requires_grad_(False)
    model.train()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    
    num_samples = 0
    total_loss = 0.0

    bg_mean = torch.tensor([0.527, 0.447, 0.403], device=args.device).view(1, 3, 1, 1)
    bg_std  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d = batch_D_d.to(args.device)
        pred_fb_d = pred_fb_d.to(args.device)
        batch_mask = ups(mask[idx]).to(args.device)

        # get random background color to replace masked pixels
        curr_batch_size = batch_D_d.size(0)
        noise = torch.normal(mean=bg_mean.expand(curr_batch_size, -1, -1, -1), 
                             std=bg_std.expand(curr_batch_size, -1, -1, -1))
        avg_val = noise.clamp(max=1, min=0)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        loss = ((t1 + t2)/len(batch_D_d))

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
            
        total_loss += loss.item() * curr_batch_size
        num_samples += curr_batch_size

    avg_loss = total_loss / num_samples

    return [avg_loss]

def evaluate_model(model, mask, train_loader, test_loader, args):

    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')

    fs_s_acc = 0
    s_count = 0
    t1_acc = 0
    mask_l0_norm = 0

    with torch.no_grad():

        for idx, batch_D_d, batch_labels, pred_fb_d in train_loader:

            batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

            background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
            background_std = 0.1*torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
            avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

            t1 = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc += torch.sum(t1).detach().cpu().item()
            fs_s = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc += torch.sum(fs_s).detach().cpu().item()
            mask_l0_norm += (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu().item()

            s_count += len(batch_labels)

    fs_t_acc = 0
    t_count = 0
    t2_acc = 0

    with torch.no_grad():
            
        for _, test_batch_D_d, test_batch_labels, fb_t in test_loader:

            fs_t = model(test_batch_D_d.to(args.device))
            fb_t = fb_t.to(args.device)

            fs_t_correct = torch.where(torch.argmax(fs_t, axis=1)==test_batch_labels.to(args.device), 1, 0)
            fs_t_acc += torch.sum(fs_t_correct).detach().cpu().item()

            t2_correct = torch.where(torch.argmax(fb_t, axis=1)==torch.argmax(fs_t, axis=1), 1, 0)
            t2_acc += torch.sum(t2_correct).detach().cpu().item()

            t_count += len(test_batch_labels)


    fs_s_acc = round(fs_s_acc/s_count, 3)
    t1_acc = round(t1_acc/s_count, 3)
    fs_t_acc = round(fs_t_acc/t_count, 3)
    t2_acc = round(t2_acc/t_count, 3)
    mask_l0_norm = round(mask_l0_norm/s_count, 3)

    print("EVAL:", time.ctime().split(" ")[3], "t1_acc:", t1_acc, "fss_acc:", fs_s_acc, "t2_acc:", t2_acc, "fst_acc:", fs_t_acc, "mask l0:", mask_l0_norm)
    return

def distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args):

    num_rounding_steps = args.r
    rounding_scheme = [0.4 - r*(0.4/num_rounding_steps) for r in range(num_rounding_steps)]
    simp_weight = [1- r*(0.9/num_rounding_steps) for r in range(num_rounding_steps)]

    # evaluate_model(model, mask, train_loader, test_loader, args)
    
    for k in range(num_rounding_steps):

        print("STEP", str(k))

        print("training mask...")
        mask_converged = False
        prev_loss, prev_prev_loss = float('inf'), float('inf')
        while (not mask_converged):

            mask_metrics = update_mask_light(mask, train_loader, model, mask_opt, simp_weight[k], args)
            mask_loss = mask_metrics[0]
            mask_converged = (mask_loss >= 0.995*prev_prev_loss) and (mask_loss <= 1.005*prev_prev_loss)
            
            prev_prev_loss = prev_loss
            prev_loss = mask_loss
            print(mask_loss)

        mask_filename = f"{args.model}_{args.dataset}_{k}_diet_mask.pth"
        torch.save(mask, os.path.join(args.model_save_path, mask_filename))
        print(f"Saved mask to {os.path.join(args.model_save_path, mask_filename)}")

        with torch.no_grad():
            mask = mask.copy_(torch.round(mask + rounding_scheme[k]))

        # evaluate_model(model, mask, train_loader, test_loader, args)

        print("training model...")
        model_converged = False
        prev_loss, prev_prev_loss = float('inf'), float('inf')
        while not model_converged:

            model_metrics = update_model_light(mask, train_loader, model, model_opt, args)
            model_loss = model_metrics[0]
            model_converged = (model_loss < 0.025) or ((model_loss >= 0.97*prev_prev_loss) and (model_loss <= 1.005*prev_prev_loss))

            prev_prev_loss = prev_loss
            prev_loss = model_loss
            print(model_loss)

        mask_converged, model_converged = False, False
        prev_loss, prev_prev_loss = float('inf'), float('inf')

        # evaluate_model(model, mask, train_loader, test_loader, args)

        model_filename = f"{args.model}_{args.dataset}_{k}_diet_model.pth"
        torch.save(model.state_dict(), os.path.join(args.model_save_path, model_filename))
        print(f"Saved model to {os.path.join(args.model_save_path, model_filename)}")

# -------------------------------------------------------- #
#                HELPER FUNCTIONS                          #
# -------------------------------------------------------- #

def get_dataloader(dataset_name, batch_size, data_root="./data"):
    # Unified transform for ResNet compatibility
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
        
    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        num_classes = 100
        
    elif dataset_name == 'HardMNIST':
        # Assumes HardMNIST data is located at data_root/hard_mnist/
        path = os.path.join(data_root, "hard_mnist")
        trainset = HardMNISTDataset(path, is_train=True, transform=transform)
        testset = HardMNISTDataset(path, is_train=False, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # For DiET, we often need access to the raw datasets later
    return trainset, testset, num_classes

def get_args():
    """Parse command-line arguments."""

    # get this file directory
    BASE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("BASE directory:", BASE)

    parser = argparse.ArgumentParser(description='PyTorch Training (DiET)')
    
    # Baseline Args
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to use')
    parser.add_argument('--model', default='resnet34', type=str, help='Model architecture')
    parser.add_argument('--data_dir', default=f'{BASE}/data', type=str, help='Path to dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--test_batch_size', default=10, type=int, help='Batch size for testing')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of workers for data loading')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--model_save_path', default=f'{BASE}/models', type=str, help='Path to save the trained model')
    parser.add_argument('--plot_path', default=f'{BASE}/pngs', type=str, help='Path to save the training plot')

    # DiET Specific Args
    parser.add_argument("--mask_lr", default=100, type=float, help="DiET: mask learning rate")
    parser.add_argument("--model_lr", default=0.0001, type=float, help="DiET: model learning rate")
    parser.add_argument("--ups", default=8, type=int, help="DiET: upsample factor")
    parser.add_argument("--r", default=5, type=int, help="DiET: number of rounding steps")
    parser.add_argument("--diet_model_path", default=None, type=str, help="Path to pre-trained model for DiET (optional, defaults to save path)")

    args = parser.parse_args()

    # ensure directories exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, f'{args.dataset}')):
        os.makedirs(os.path.join(args.data_dir, f'{args.dataset}'))
    args.dataset_dir = os.path.join(args.data_dir, f'{args.dataset}')
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
    
    # Hardcoded for DiET consistency
    args.im_size = 224 
    
    return args

def get_model(num_classes, args, freeze_first=False):
    """Load pre-trained model and adapt it for the dataset."""
    model = getattr(torchvision.models, args.model)(weights="DEFAULT")

    if freeze_first:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# -------------------------------------------------------- #
#               BASELINE TRAINING LOGIC                    #
# -------------------------------------------------------- #

def train(epoch, model, trainloader, optimizer, loss_fn, device):
    """Train the model for one epoch."""
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred_c = pred.max(1)
        total += y.size(0)
        correct += pred_c.eq(y).sum().item()

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    return train_loss / len(trainloader), 100. * correct / total

def test(model, testloader, loss_fn, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            _, pred_c = pred.max(1)
            total += y.size(0)
            correct += pred_c.eq(y).sum().item()
    
    acc = 100.*correct/total
    print(f'Test Loss: {test_loss/(len(testloader)):.3f} | Acc: {acc:.3f}% ({correct}/{total})')
    return test_loss / len(testloader), acc

def plot_metrics(train_losses, train_accs, test_losses, test_accs, save_path):
    """Plot and save training metrics."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'ro-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Plot saved to {save_path}')

# -------------------------------------------------------- #
#                  PREDICTION HELPER                       #
# -------------------------------------------------------- #

def get_predictions_from_dataset(model, dataset, args):
    """
    Returns the input model's predictions on the full dataset.
    This replaces the DiET 'get_predictions' to work with baseline datasets.
    """
    # Create a loader that does NOT shuffle, so indices align
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    preds = torch.zeros((len(dataset), args.num_classes))
    sm = torch.nn.Softmax(dim=1)
    
    print("Calculating Teacher Predictions...", flush=True)
    model.eval()
    with torch.no_grad():
        current_idx = 0
        for imgs, _ in loader:
            imgs = imgs.to(args.device)
            output = sm(model(imgs)).cpu()
            
            # fill tensor
            batch_len = output.size(0)
            preds[current_idx : current_idx + batch_len] = output
            current_idx += batch_len
            
            if current_idx % 1000 == 0:
                print(f"Preds: {current_idx}/{len(dataset)}")
                
    return preds

# -------------------------------------------------------- #
#                       MAIN                               #
# -------------------------------------------------------- #

def main():
    """Main function to run the training script."""
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device # Add to args for DiET compatibility
    print(f"Using device: {device}")

    # Load Data (Returns Dataset Objects, not loaders yet)
    trainset, testset, num_classes = get_dataloader(args.dataset, args.batch_size, data_root=args.data_dir)
    print(f'Number of classes: {num_classes}')
    args.num_classes = num_classes # For DiET

    # Initialize Model
    model = get_model(num_classes, args)
    model = model.to(device)

    print("------------------------------------------------")
    print("STARTING DiET TRAINING")
    print("------------------------------------------------")
    
    # Load the pre-trained baseline model weights
    load_path = args.diet_model_path if args.diet_model_path else os.path.join(args.model_save_path, f'{args.model}_{args.dataset}_model.pth')
    if os.path.exists(load_path):
        print(f"Loading teacher model from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print(f"WARNING: Model file not found at {load_path}. Using random weights (Not recommended for DiET).")
        raise RuntimeError()

    model.eval()

    # 1. Get Teacher Predictions
    train_preds = get_predictions_from_dataset(model, trainset, args)
    test_preds = get_predictions_from_dataset(model, testset, args)

    # 2. Wrap Datasets for DiET (yields idx, img, label, pred)
    train_diet_dataset = DiETDatasetWrapper(trainset, train_preds)
    test_diet_dataset = DiETDatasetWrapper(testset, test_preds)

    train_loader = torch.utils.data.DataLoader(train_diet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_diet_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Loaded Data for DiET.")

    # 3. Setup Mask and Optimizers
    # Note: Mask size is based on im_size/ups. Assumes 224x224 images.
    mask = torch.ones((len(trainset), 1, args.im_size//args.ups, args.im_size//args.ups))
    mask = mask.requires_grad_(True)
    
    mask_opt = torch.optim.SGD([mask], lr=args.mask_lr)
    mask_opt.zero_grad()
    
    model_opt = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    model_opt.zero_grad()

    # 4. Run DiET
    distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args)
    
    print("DiET Training Finished.")
    return

if __name__ == '__main__':
    main()