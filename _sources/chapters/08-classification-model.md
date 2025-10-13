# Building Classification Models

## Introduction

In this chapter, we'll build Convolutional Neural Network (CNN) models to classify FCAL showers as either photon showers or splitoff events. We'll progress from simple architectures to more sophisticated models.

## Problem Formulation

### Binary Classification Task

**Goal**: Given an FCAL shower image, predict whether it's a single photon or contains splitoffs.

**Input**: 
- 2D image (e.g., 32×32 or 64×64 pixels)
- Pixel values represent energy deposition in FCAL blocks
- May include additional channels (time, normalized energy)

**Output**:
- Probability P(photon | image)
- Classification: photon if P > threshold (typically 0.5), splitoff otherwise

**Loss Function**: Binary cross-entropy
```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```
where y ∈ {0,1} is true label, ŷ is predicted probability

## CNN Architecture Design

### Why CNNs for FCAL Data?

CNNs are ideal for FCAL shower classification because:

1. **Spatial Structure**: FCAL showers have spatial patterns that CNNs naturally capture
2. **Translation Invariance**: Showers can occur anywhere in FCAL; convolutions are translation-equivariant
3. **Hierarchical Features**: CNNs learn features from local patterns (edges) to global structures (shower shape)
4. **Parameter Efficiency**: Shared weights reduce parameters compared to fully connected networks

### Building Blocks

**Convolutional Layer:**
```python
Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```
- Learns spatial filters to detect features
- kernel_size typically 3×3 or 5×5
- padding='same' preserves spatial dimensions

**Activation Function:**
```python
ReLU() or LeakyReLU()
```
- Introduces non-linearity
- ReLU(x) = max(0, x)
- LeakyReLU allows small negative gradient

**Pooling Layer:**
```python
MaxPool2d(kernel_size=2, stride=2)
```
- Reduces spatial dimensions by factor of 2
- Provides translation invariance
- Reduces computational cost

**Batch Normalization:**
```python
BatchNorm2d(num_features)
```
- Normalizes layer inputs
- Stabilizes training
- Acts as regularization

**Dropout:**
```python
Dropout(p=0.5)
```
- Randomly zeros fraction p of neurons during training
- Reduces overfitting
- Not used during inference

**Fully Connected Layer:**
```python
Linear(in_features, out_features)
```
- Standard neural network layer
- Used at end for classification

## Model 1: Simple CNN Baseline

### Architecture

Let's start with a simple 3-layer CNN:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers (adjust based on input size)
        # For 32×32 input: after 3 pooling layers -> 4×4×64 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # 32×32 -> 16×16
        
        # Conv block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # 16×16 -> 8×8
        
        # Conv block 3: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # 8×8 -> 4×4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Instantiate model
model = SimpleCNN(input_channels=1, num_classes=1)
print(model)
```

### Model Summary

```
SimpleCNN(
  (conv1): Conv2d(1, 16, kernel_size=(5, 5), padding=(2, 2))
  (conv2): Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(1024, 128)
  (fc2): Linear(128, 1)
  (dropout): Dropout(p=0.5)
)
Total parameters: ~150K
```

### Training Setup

```python
import torch.optim as optim

# Loss function
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_simple_cnn.pth')
        print('Model saved!')
    print()
```

## Model 2: Improved CNN with Batch Normalization

### Architecture

Adding batch normalization for better training stability:

```python
class ImprovedCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(ImprovedCNN, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### Key Improvements

1. **Batch Normalization**: After each conv layer for stable training
2. **Deeper Network**: 4 conv blocks instead of 3
3. **Global Average Pooling**: Reduces parameters, prevents overfitting
4. **More Filters**: 32 -> 64 -> 128 -> 256 progression

Expected performance: ~92-95% validation accuracy

## Model 3: ResNet-Inspired Architecture

### Residual Blocks

ResNets use skip connections to enable deeper networks:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = F.relu(out)
        
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(ResNetClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
```

Expected performance: ~95-97% validation accuracy

## Model 4: Multi-Input Architecture

### Using Additional Features

Combine CNN with kinematic features:

```python
class MultiInputCNN(nn.Module):
    def __init__(self, input_channels=1, num_kinematic_features=4, num_classes=1):
        super(MultiInputCNN, self).__init__()
        
        # CNN for image processing
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # MLP for kinematic features
        self.fc_kin1 = nn.Linear(num_kinematic_features, 32)
        self.fc_kin2 = nn.Linear(32, 64)
        
        # Combined classifier
        self.fc1 = nn.Linear(128 + 64, 128)  # CNN features + kinematic features
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, image, kinematics):
        # Process image
        x = F.relu(self.bn1(self.conv1(image)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Process kinematics
        k = F.relu(self.fc_kin1(kinematics))
        k = F.relu(self.fc_kin2(k))
        
        # Combine
        combined = torch.cat([x, k], dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        output = self.fc2(combined)
        
        return output
```

**Kinematic features** might include:
- Total cluster energy
- Number of blocks in cluster
- Cluster RMS (spread)
- Energy asymmetry

## Hyperparameter Tuning

### Key Hyperparameters

**Architecture:**
- Number of layers: 3-6 conv layers typically optimal
- Filters per layer: Powers of 2 (32, 64, 128, 256)
- Kernel size: 3×3 or 5×5
- Dropout rate: 0.3-0.5

**Training:**
- Learning rate: 1e-3 to 1e-4 (Adam)
- Batch size: 32-128 (depends on GPU memory)
- Epochs: 50-100 (with early stopping)
- Weight decay: 1e-5 to 1e-4 (L2 regularization)

### Grid Search Example

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [32, 64, 128],
    'dropout': [0.3, 0.5],
    'weight_decay': [1e-5, 1e-4]
}

best_val_acc = 0
best_params = None

for params in ParameterGrid(param_grid):
    model = ImprovedCNN()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Train model (abbreviated)
    val_acc = train_and_evaluate(model, optimizer, params)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params
        
print(f'Best validation accuracy: {best_val_acc:.4f}')
print(f'Best parameters: {best_params}')
```

## Practical Tips

### 1. Start Simple
- Begin with baseline model
- Verify training works before adding complexity
- Establish performance benchmark

### 2. Monitor Training
```python
# Use tensorboard or similar
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
```

### 3. Regularization
- Use dropout (0.5 common for FC layers)
- Apply batch normalization
- Add weight decay (L2 regularization)
- Use data augmentation (covered in next chapter)

### 4. Debug Overfitting
**Symptoms**: Train accuracy >> Val accuracy

**Solutions**:
- Increase dropout
- Add data augmentation
- Reduce model capacity
- Collect more training data
- Add L2 regularization

### 5. Debug Underfitting
**Symptoms**: Low train and val accuracy

**Solutions**:
- Increase model capacity (more layers/filters)
- Train longer
- Reduce regularization
- Check data quality
- Verify loss function and labels

## Model Comparison

| Model | Parameters | Val Accuracy | Training Time |
|-------|-----------|--------------|---------------|
| SimpleCNN | ~150K | 89-91% | Fast |
| ImprovedCNN | ~500K | 92-95% | Medium |
| ResNet | ~1M | 95-97% | Slower |
| MultiInput | ~600K | 94-96% | Medium |

Choose based on:
- **SimpleCNN**: Quick baseline, limited data
- **ImprovedCNN**: Best balance of performance/complexity
- **ResNet**: Maximum accuracy, large dataset
- **MultiInput**: When kinematic features are available

## Summary

Key points:
1. CNNs naturally capture spatial patterns in FCAL showers
2. Start simple, iterate to improve
3. Batch normalization stabilizes training
4. Residual connections enable deeper networks
5. Multi-input models can leverage additional features
6. Monitor training to diagnose issues

## Next Steps

- [Training and Optimization](09-training.md) - Advanced training techniques
- [Model Evaluation](10-evaluation.md) - Comprehensive performance assessment
- Practice: Implement and train these models on your FCAL dataset!
