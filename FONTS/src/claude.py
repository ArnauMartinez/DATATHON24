import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import os
from sklearn.metrics import confusion_matrix

def set_seed(seed: int = 123):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MetricsTracker:
    """Tracks and computes various training metrics."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total = 0
        self.correct = 0
        self.running_loss = 0.0
        self.predictions = []
        self.targets = []
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, loss: float):
        pred = outputs.argmax(dim=1)
        correct = pred.eq(targets)
        
        self.total += targets.size(0)
        self.correct += correct.sum().item()
        self.running_loss += loss * targets.size(0)
        
        self.predictions.extend(pred.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        for label in range(self.num_classes):
            mask = targets == label
            if mask.sum() > 0:
                self.class_correct[label] += correct[mask].sum().item()
                self.class_total[label] += mask.sum().item()
    
    def get_metrics(self) -> Dict[str, float]:
        accuracy = (self.correct / self.total) * 100
        avg_loss = self.running_loss / self.total
        
        class_accuracies = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.class_total[i] > 0:
                class_accuracies[i] = (self.class_correct[i] / self.class_total[i]) * 100
        
        conf_matrix = confusion_matrix(self.targets, self.predictions)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'per_class_accuracy': class_accuracies.tolist(),
            'confusion_matrix': conf_matrix
        }

class ConvBlock(nn.Module):
    """Improved Convolutional Block with BatchNorm and Dropout."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ImprovedNet(nn.Module):
    """Improved Neural Network Architecture."""
    def __init__(self, num_classes: int = 142, dropout: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 64, 3),
            ConvBlock(64, 128, 3, stride=2),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 512, 3, stride=2),
            ConvBlock(512, 512, 3)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_transforms(is_training: bool = True):
    """Get data transforms for training or evaluation."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def train_epoch(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        metrics_tracker: MetricsTracker,
        log_interval: int = 10
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics_tracker.reset()
    
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        metrics_tracker.update(output, target, loss.item())
        
        if batch_idx % log_interval == 0:
            print(f'Training: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    return metrics_tracker.get_metrics()

@torch.no_grad()
def evaluate(
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        metrics_tracker: MetricsTracker
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    metrics_tracker.reset()
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        metrics_tracker.update(output, target, loss.item())
    
    metrics = metrics_tracker.get_metrics()
    print(f'\nTest set: Average loss: {metrics["loss"]:.4f}, '
          f'Accuracy: {metrics["accuracy"]:.2f}%')
    
    return metrics

def plot_metrics(train_history: List[Dict], val_history: List[Dict]):
    """Plot training and validation metrics."""
    epochs = range(1, len(train_history) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [m['accuracy'] for m in train_history], label='Train')
    plt.plot(epochs, [m['accuracy'] for m in val_history], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [m['loss'] for m in train_history], label='Train')
    plt.plot(epochs, [m['loss'] for m in val_history], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    num_classes: int,
    initial_lr: float = 0.01,
    weight_decay: float = 1e-4
) -> Tuple[List[Dict], List[Dict]]:
    """Complete training pipeline."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, 
                               momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    train_metrics_tracker = MetricsTracker(num_classes)
    val_metrics_tracker = MetricsTracker(num_classes)
    
    train_history = []
    val_history = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        train_metrics = train_epoch(
            train_loader, model, optimizer, criterion,
            device, train_metrics_tracker
        )
        train_history.append(train_metrics)
        
        # Validation phase
        val_metrics = evaluate(
            val_loader, model, criterion,
            device, val_metrics_tracker
        )
        val_history.append(val_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Plot progress
        plot_metrics(train_history, val_history)
    
    return train_history, val_history

if __name__ == "__main__":
    # Setup
    set_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    hparams = {
        'batch_size': 32,
        'num_epochs': 30,
        'initial_lr': 0.01,
        'weight_decay': 1e-4,
        'num_classes': 142
    }
    
    # Data loading
    trainset = datasets.ImageFolder(
        root="/Users/laura/DATATHON24/DATATHON24/data/archive/images",
        transform=get_transforms(is_training=True)
    )
    
    # Limit dataset size if needed
    trainset.samples = trainset.samples[:10000]
    trainset.targets = trainset.targets[:10000]
    
    train_loader = DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        datasets.ImageFolder(
            root="/Users/laura/DATATHON24/DATATHON24/data/archive/images",
            transform=get_transforms(is_training=False)
        ),
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create and train model
    model = ImprovedNet(num_classes=hparams['num_classes']).to(device)
    
    train_history, val_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=hparams['num_epochs'],
        device=device,
        num_classes=hparams['num_classes'],
        initial_lr=hparams['initial_lr'],
        weight_decay=hparams['weight_decay']
    )