import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt

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
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
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

# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
def prepare_data(args):
    """Download and prepare dataset."""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    trainset = getattr(torchvision.datasets, args.dataset)(root=args.dataset_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = getattr(torchvision.datasets, args.dataset)(root=args.dataset_dir, train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    
    return trainloader, testloader

def get_model(num_classes, args):
    """Load pre-trained model and adapt it for the dataset."""
    model = getattr(torchvision.models, args.model)(weights="DEFAULT")
    
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

    trainloader, testloader = prepare_data(args)
    
    # get number of classes from dataset
    num_classes = len(trainloader.dataset.classes)
    print(f'Number of classes: {num_classes}')

    model = get_model(num_classes, args)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, loss_fn, device)
        test_loss, test_acc = test(model, testloader, loss_fn, device)
        # scheduler.step()

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
