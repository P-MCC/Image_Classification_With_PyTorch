import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the SimpleCNN class for image classification.

        Args:
            num_classes (int): The number of output classes for classification.
        """
        super(SimpleCNN, self).__init__()

        # Convolutional Layer 1: Input channels (3 for RGB), 16 output channels, 3x3 kernel size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer with 2x2 filter
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Conv layer 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 32 channels with 32x32 feature map size
        self.fc2 = nn.Linear(128, num_classes)   # Final output layer

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1, ReLU, and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2, ReLU, and pooling
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, flattened_size)

        # Fully connected layers with ReLU and output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x