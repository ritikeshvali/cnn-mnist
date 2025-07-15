# PyTorch MNIST CNN Classifier

A Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Features
- CNN architecture with 2 convolutional layers
- MaxPooling for dimensionality reduction
- MNIST dataset training and testing
- GPU support with CUDA
- Accuracy evaluation on training and test sets

## Requirements
```bash
pip install torch torchvision
```

## Usage
```bash
python cnn.py
```

## Model Architecture
### CNN Model
- **Conv Layer 1**: 1 → 8 channels, 3x3 kernel, ReLU activation
- **MaxPool**: 2x2 kernel, stride 2
- **Conv Layer 2**: 8 → 16 channels, 3x3 kernel, ReLU activation  
- **MaxPool**: 2x2 kernel, stride 2
- **Fully Connected**: 16×7×7 → 10 output classes

### Alternative NN Model (included)
- Input layer: 784 neurons (28×28 flattened images)
- Hidden layer: 50 neurons with ReLU activation
- Output layer: 10 neurons (digit classes)

## Training Parameters
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Optimizer: Adam
- Loss function: CrossEntropyLoss