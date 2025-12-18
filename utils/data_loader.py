import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

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
        img = self.transform(img)
        return img, label

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

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, num_classes