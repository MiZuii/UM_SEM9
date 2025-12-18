import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt

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




def get_args():
    """Parse command-line arguments."""

    # get this file directory
    BASE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("BASE directory:", BASE)

    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to use (default: CIFAR10)')
    parser.add_argument('--model', default='resnet34', type=str, help='Model architecture (default: resnet34)')
    parser.add_argument('--data_dir', default=f'{BASE}/data', type=str, help='Path to dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--test_batch_size', default=100, type=int, help='Batch size for testing')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--model_save_path', default=f'{BASE}/models', type=str, help='Path to save the trained model')
    parser.add_argument('--plot_path', default=f'{BASE}/pngs', type=str, help='Path to save the training plot')

    args = parser.parse_args()

    # ensure direcotires exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(os.path.join(args.data_dir, f'{args.dataset}')):
        os.makedirs(os.path.join(args.data_dir, f'{args.dataset}'))

    args.dataset_dir = os.path.join(args.data_dir, f'{args.dataset}')

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
    
    return args

def get_model(num_classes, args):
    """Load pre-trained model and adapt it for the dataset."""
    model = getattr(torchvision.models, args.model)(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

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

def main():
    """Main function to run the training script."""
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader, num_classes = get_dataloader(args.dataset, args.batch_size, data_root=args.data_dir)
    print(f'Number of classes: {num_classes}')

    model = get_model(num_classes, args)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # -------------------------------------------------------- #
    #                          WARMUP                          #
    # -------------------------------------------------------- #

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    warmup_epochs = 3
    for epoch in range(warmup_epochs):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, loss_fn, device)
        test_loss, test_acc = test(model, testloader, loss_fn, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # -------------------------------------------------------- #
    #                        FINE TUNING                       #
    # -------------------------------------------------------- #

    for param in model.parameters():
        param.requires_grad = True

    finetune_lr = args.lr * 0.1 
    optimizer = optim.AdamW(model.parameters(), lr=finetune_lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, loss_fn, device)
        test_loss, test_acc = test(model, testloader, loss_fn, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print('Finished Training')

    torch.save(model.state_dict(), os.path.join(args.model_save_path, f'{args.model}_{args.dataset}_model.pth'))
    print(f'Model saved to {os.path.join(args.model_save_path, f'{args.model}_{args.dataset}_model.pth')}')

    plot_metrics(train_losses, train_accs, test_losses, test_accs, os.path.join(args.plot_path, f'{args.model}_{args.dataset}_training_plot.png'))

if __name__ == '__main__':
    main()
