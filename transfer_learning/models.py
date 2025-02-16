import torch.nn as nn
from torchvision import models


def freeze_top_layers(model, num_layers):
    """Freeze the first layers of the model.

    Mitigate overfitting and because they extract more generic features
    """
    count = 0
    for child in model.children():
        count += 1
        if count < num_layers:
            for param in child.parameters():
                param.requires_grad = False
    return model


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # First convolutional layer:
        # Accepts 3-channel (RGB) input images and produces 32 feature maps
        # Kernel size is 4x4 with a stride of 1 and no padding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        # Batch normalization to stabilize and accelerate training
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer:
        # Takes 32 input feature maps and produces 64 output feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        # Batch normalization for the second layer
        self.bn2 = nn.BatchNorm2d(64)
        # Third convolutional layer:
        # Increases the number of feature maps to 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        # Batch normalization for the third layer
        self.bn3 = nn.BatchNorm2d(128)
        # Fourth convolutional layer:
        # Produces 254 output feature maps
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        # Batch normalization for the Fourth layer
        self.bn4 = nn.BatchNorm2d(256)

        # Max pooling layer to down-sample the feature maps
        # Reduces the spatial dimensions of the feature maps
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        # Fully connected layer:
        # Takes the flattened output from the previous layers and maps it to 512 neurons
        self.fc1 = nn.Linear(9*256, 512)
        # Output layer for classification
        self.fc2 = nn.Linear(512, num_classes)
        # Flatten layer to convert multi-dimensional input to 1D
        self.flatten = nn.Flatten()
        # ReLU activation function for non-linearity
        self.relu = nn.ReLU()
        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.bn1(self.conv1(x)))  # First convolution + ReLU + BatchNorm
        x = self.pool(x)  # Max pooling
        x = self.relu(self.bn2(self.conv2(x)))  # Second convolution + ReLU + BatchNorm
        x = self.pool(x)  # Max pooling
        x = self.relu(self.bn3(self.conv3(x)))  # Third convolution + ReLU + BatchNorm
        x = self.pool(x)  # Max pooling
        x = self.relu(self.bn4(self.conv4(x)))  # Fourth convolution + ReLU + BatchNorm
        x = self.flatten(x)  # Flatten the output
        x = self.relu(self.fc1(x))  # First fully connected layer + ReLU
        x = self.dropout(x)   # Apply dropout
        x = self.fc2(x)  # Final output layer for classification
        return x  # Return the class scores


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_layers=0):
        super().__init__()
        self.resnet18 = models.resnet18(weights=pretrained)
        if freeze_layers > 0:
            # Freeze top layers
            self.resnet18 = freeze_top_layers(self.resnet18, freeze_layers)
        else:
            pass
        # replace the last layer
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x
