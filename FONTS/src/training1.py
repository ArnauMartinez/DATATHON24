import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from torchvision import transforms
import os

seed = 123
random.seed(seed)
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Let's define some hyper-parameters
hparams = {
    'batch_size': 100,
    'num_epochs': 10,
    'val_batch_size': 100,
    'num_classes': 142,
    'learning_rate': 0.1,
    'log_interval': 100,
}

# Dataset class
class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: str, label_file: str, transform=None):
        self.image_folder = image_folder
        self.label_df = pd.read_csv(label_file)  # Read CSV file with labels
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx, 0]
        label = list(self.label_df.iloc[idx, 1:].values)
        label = torch.Tensor(label)

        img_path = self.image_folder + img_name

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# Define ConvBlock first so it is available when defining BigNet
class ConvBlock(nn.Module):
    def __init__(self, num_inp_channels: int, num_out_fmaps: int, kernel_size: int, stride: int=1):
        super().__init__()
        self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


# Define the BigNet model
class BigNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 512, kernel_size=3, stride=4)
        self.conv2 = ConvBlock(512, 1024, kernel_size=3, stride=4)
        self.mlp = nn.Sequential(
            nn.Linear(143360, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 142),  # This should match the number of classes in your dataset
            nn.Sigmoid()  # Use Sigmoid activation instead of LogSoftmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.mlp(x)


# Define a function to compute accuracy
def compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    # If target is multi-label, compute accuracy based on threshold (0.5 here)
    pred = output > 0.5
    correct = (pred == target).sum().item()
    accuracy = correct / target.numel()
    return accuracy * 100  # Return percentage


# Train epoch function
def train_epoch(
        train_loader: DataLoader,
        network: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        log_interval: int,
        device: torch.device
) -> Tuple[float, float]:
    network.train()

    train_loss = []
    acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0 or batch_idx >= len(train_loader):
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_acc = 100. * acc / len(train_loader.dataset)
    return np.mean(train_loss), avg_acc


# Evaluation epoch function
@torch.no_grad()
def eval_epoch(
        test_loader: DataLoader,
        network: nn.Module,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    network.eval()

    test_loss = 0
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = network(data)

        # Apply the loss function and accumulate
        test_loss += criterion(output, target).item()

        # Calculate the number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    test_loss /= len(test_loader)
    test_acc = 100. * acc / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc}/{len(test_loader.dataset)} '
          f'({test_acc:.0f}%)\n')
    
    return test_loss, test_acc


# Train the model for multiple epochs
def train_net(
        network: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        plot: bool=True,
        log_interval: int=10
) -> Dict[str, List[float]]:
    """ Function that trains and evaluates a model for num_epochs, 
        showing a plot of losses and accuracies and returning them.
    """
    tr_losses = []
    tr_accs = []
    te_losses = []
    te_accs = []

    network.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(train_loader, network, optimizer, criterion, epoch, log_interval, device)
        te_loss, te_acc = eval_epoch(eval_loader, network, criterion, device)
        te_losses.append(te_loss)
        te_accs.append(te_acc)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
    
    rets = {'tr_losses': tr_losses, 'te_losses': te_losses,
            'tr_accs': tr_accs, 'te_accs': te_accs}
    
    if plot:
        # Plotting loss and accuracy over epochs
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('NLLLoss')
        plt.plot(tr_losses, label='train')
        plt.plot(te_losses, label='eval')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Eval Accuracy [%]')
        plt.plot(tr_accs, label='train')
        plt.plot(te_accs, label='eval')
        plt.legend()

    return rets


# Now, setting up the data paths and transformations
image_folder = "/Users/laura/DATATHON24/DATATHON24/data/archive/images/images/"
label_file = "/Users/laura/DATATHON24/DATATHON24/FONTS/src/images_targets.csv"

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor
])

# Create the dataset
dataset = ImageLabelDataset(image_folder=image_folder, label_file=label_file, transform=transform)

# Splitting the dataset into train, eval, and test
train_size = int(0.6 * len(dataset))
eval_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - eval_size
trainset, evalset, testset = random_split(dataset, [train_size, eval_size, test_size])

# Create DataLoaders
train_loader = DataLoader(trainset, batch_size=1000, shuffle=True)
eval_loader = DataLoader(evalset, batch_size=1000, shuffle=False)
test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

# Initialize the model
model = BigNet()

# Create optimizer (Adam in this case)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Set device (e.g., 'cuda' or 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
bigmodel_log = train_net(model, train_loader, eval_loader, optimizer, num_epochs=4, device=device)
