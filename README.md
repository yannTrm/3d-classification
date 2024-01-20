# Image Classification with PyTorch

## Overview

This project demonstrates a simple image classification task using PyTorch. The code includes the following components:

### Requirements

Ensure you have the necessary dependencies installed:

```bash
pip install torch torchvision matplotlib tqdm scikit-learn
```

## Configuration

Set the correct path to the parent directory containing the src folder, and choose the device (CPU or GPU).

```python
import os
import sys
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append('..')  # Replace '..' with the actual path to the parent directory containing 'src'.

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
```

## Dataset and Model

Import the necessary modules and classes for the custom dataset (CustomDataset), dataset splitting function (split_dataset_1), and classifier model (Classifier).

```python
from objects.dataset import CustomDataset
from objects.utils import split_dataset_1
from objects.models import Classifier
```


### Example Usage

Load and preprocess the dataset, splitting it into training and testing sets.

```python
# Example usage:
root_directory = '../2d_data'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

custom_dataset = CustomDataset(root_directory, transform=transform)

train_set, test_set = split_dataset_1(custom_dataset, test_size=0.2)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
```

### Model Training

Initialize the classifier, define the loss function, optimizer, and set the number of training epochs.


```python
# Initialize the classifier
num_classes = len(custom_dataset.classes)
classifier = Classifier(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Set the number of training epochs
num_epochs = 10
```

### Training Loop

Train the model and evaluate on the test set.

```python
# Training and testing loop
for epoch in range(num_epochs):
    # Training phase
    classifier.train()  # Set the model to training mode
    running_loss = 0.0
    predictions = []
    true_labels = []

    for images, labels in tqdm(train_loader):
        # Training steps...

    # Testing phase
    classifier.eval()  # Set the model to evaluation mode
    test_running_loss = 0.0
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        # Testing steps...
```

### Model Evaluation

Evaluate the model on the test set and print the accuracy.


```python
# Evaluate the model on the test set
classifier.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    # Evaluation steps...

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
```

### Learning Curves

Visualize the learning curves (training and test loss, training and test accuracy) using matplotlib.

```python
# Plot the learning curves
plt.plot(train_loss_values, label='Training Loss')
plt.plot(test_loss_values, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show()

plt.plot(train_accuracy_values, label='Training Accuracy')
plt.plot(test_accuracy_values, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()
```

### Conclusion

Customize the code to suit your specific use case. Experiment with different architectures, hyperparameters, and augmentation techniques for better performance. Happy coding!
