# Image Place Classification

![Image Classification Logo](Image_Classifixation.png)

<!-- Replace the path above with the actual path to your image file -->

A deep learning project for classifying images into different place categories using PyTorch and CNN architecture.

## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification tasks. The model is designed to classify images into 6 different categories, using a custom CNN architecture with modern deep learning techniques.

## Model Architecture

The model uses a custom CNN architecture (`BetterCNN`) with the following features:
- Multiple convolutional layers with increasing channels (32 → 64 → 128)
- Batch Normalization for better training stability
- ReLU activation functions
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Adaptive Average Pooling for flexible input sizes
- Fully connected layers for final classification

## Image Classes

The model is trained to classify images into 6 different categories:
1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

## Visualization Tools

The project includes visualization tools to help analyze the model's predictions:

1. Random Image Visualization:
```python
# Function to visualize random images with class distribution
def visualize_random_images(num_images=5):
    # Set random seed based on current time
    torch.manual_seed(int(time.time()))
    
    # Generate random indices
    random_indices = torch.randperm(len(images))[:num_images]
    
    # Track class distribution
    class_counts = {}
    
    # Visualize images
    for i in random_indices:
        class_name = train_dataset.classes[labels[i]]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        imshow(images[i])
        print(f'Predicted: {train_dataset.classes[preds[i]]} | True: {class_name}')
    
    # Print class distribution
    print("\nClass distribution in selected images:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
```

2. One Image Per Class Visualization:
```python
# Function to get one image from each class
def visualize_one_per_class():
    # Create a dictionary to store indices for each class
    class_indices = {}
    
    # Find one random image from each class
    for idx, (_, label) in enumerate(zip(images, labels)):
        class_name = train_dataset.classes[label]
        if class_name not in class_indices:
            class_indices[class_name] = idx
            if len(class_indices) == len(train_dataset.classes):
                break
    
    # Visualize one image from each class
    print("Showing one image from each class:")
    for class_name, idx in class_indices.items():
        imshow(images[idx])
        print(f'Class: {class_name}')
        print(f'Predicted: {train_dataset.classes[preds[idx]]} | True: {class_name}\n')
```

These visualization tools help in:
- Verifying model predictions
- Ensuring balanced class representation
- Analyzing model performance across different categories

## Requirements

- Python 3.10 or higher
- PyTorch 2.1.0 or higher
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Image_Place_classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset:
   - Organize your images in appropriate directories
   - Ensure your images are properly labeled

2. Train the model:
```python
from model import BetterCNN
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BetterCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
```

## Model Structure

The model consists of two main parts:

1. Feature Extraction (`self.features`):
   - Three convolutional blocks
   - Each block contains:
     - Two convolutional layers
     - Batch normalization
     - ReLU activation
     - Max pooling
     - Dropout

2. Classifier (`self.classifier`):
   - Adaptive average pooling
   - Flattening layer
   - Two fully connected layers
   - Batch normalization
   - ReLU activation
   - Dropout

## Performance

The model is designed to achieve good performance on image classification tasks with:
- Efficient feature extraction through convolutional layers
- Regularization through dropout and batch normalization
- Flexible input size handling through adaptive pooling
