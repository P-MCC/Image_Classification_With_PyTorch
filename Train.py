import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from CustomDataLoader import PlayingCardDataset
from Model import SimpleCNN

data_path ='M:/Datasets/Playing_Card'
train_dir = '/train'
valid_dir = '/valid'
test_dir = '/test'
# Assuming your dataset is already defined and ready
# Replace 'PlayingCardDataset' and 'data_dir' with your actual dataset and path
train_dataset = PlayingCardDataset(data_dir=data_path+train_dir, transform=True)
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
num_classes = len(train_dataset.classes)  # Get the number of classes from the dataset
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Check if CUDA (GPU) is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Training loop
num_epochs = 10 # Number of epochs to train

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        # Move images and labels to the device (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Training complete')
