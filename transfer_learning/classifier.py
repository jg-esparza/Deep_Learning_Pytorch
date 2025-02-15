import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from datetime import datetime

from data_utils import mean, std, load_dataset, load_batch, getting_total_batch_sizes, getting_classes_names, to_dict
from training_utils import train_model, evaluate_model, save_model, load_model


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


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def getting_number_of_classes(dataset):
    """Return number of classes in dataset."""
    return len(getting_classes_names(dataset))


def getting_data_batches(train_dir, test_dir, batch_size): # noqa
    """Return batches of training and testing datasets in a dictionary.

    Load training and testing data from the provided directories,
    Use dataloaders to get batches of training and testing data.
    """
    train_data = load_dataset(root=train_dir, transform=transforms)
    test_data = load_dataset(root=test_dir, transform=transforms)
    train_loader = load_batch(train_data, batch_size=batch_size, shuffle=True)
    test_loader = load_batch(test_data, batch_size=batch_size, shuffle=False)
    return to_dict(train_loader, test_loader)


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

def training_classifier(model, dataloaders, num_epochs, batch_size, total_batch_sizes, path = None):  # noqa
    """Training a classifier with predefined criterion, optimizer and learning rate scheduler.

    Saves the model if provides a directory.
    """
    # define a loss function
    criterion = nn.CrossEntropyLoss()
    # define optimizer
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # define learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                        total_batch_sizes, batch_size, num_epochs=num_epochs)

    if path is not None:
        save_model(model, path)
    return model


def evaluating_classifier(model, dataloaders, model_path=None):
    """Evaluates a classifier using testing data.

    Loads a trained model if provides a directory.
    """
    if model_path is not None:
        load_model(model, model_path)
    evaluate_model(model, dataloaders['test'])


def run_classifier(model, num_epochs, batch_size, data_batches, saving_path = None):  # noqa
    """Train and evaluate a classifier."""
    total_batch_sizes = getting_total_batch_sizes(data_batches['train'], data_batches['test'])
    print('Total batch sizes: {}'.format(total_batch_sizes))
    start_time = datetime.now()
    model = training_classifier(model, data_batches, num_epochs, batch_size, total_batch_sizes, path=saving_path)
    end_time = datetime.now()
    print('Total training time: {}'.format(end_time - start_time))
    evaluating_classifier(model, data_batches, saving_path)


if __name__ == '__main__':
    epochs = 5
    batch_size = 8
    train_dir = '../datasets/brain_tumor/train'
    test_dir = '../datasets/brain_tumor/test'
    data_batches = getting_data_batches(train_dir, test_dir, batch_size)

    custom_cnn = CustomCNN(num_classes=4)
    print('Custom CNN: {}'.format(custom_cnn))
    run_classifier(custom_cnn, epochs, batch_size, data_batches, '../models/Custom_CNN.pth')
    # evaluating_classifier(custom_cnn, data_batches, model_path='../models/Custom_CNN.pth')

    # VGG16 freeze top layers
    vgg16_model = models.vgg16(weights=True)
    vgg16_model = freeze_top_layers(vgg16_model, 1)
    # replace the last layer
    num_ftrs = vgg16_model.classifier[6].in_features
    vgg16_model.fc = nn.Linear(num_ftrs, 4)
    print('VGG16_freezing top layers: {}'.format(vgg16_model))
    saving_path_VGG16 = '../models/VGG16_freeze_top_layers.pth'
    run_classifier(vgg16_model, epochs, batch_size, data_batches, saving_path=saving_path_VGG16)
    # evaluating_classifier(vgg16_model, data_batches, model_path=saving_path_VGG16)

    # Resnet18 freeze top layers
    resnet18_model = models.resnet18(pretrained=True)
    resnet18_model = freeze_top_layers(resnet18_model, 6)
    # replace the last layer
    num_ftrs = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(num_ftrs, 4)
    print('Resnet18_freezing top layers: {}'.format(resnet18_model))
    saving_path_resnet = '../models/Resnet18_freeze_top_layers.pth'
    run_classifier(resnet18_model, epochs, batch_size, data_batches, saving_path=saving_path_resnet)
    # evaluating_classifier(resnet18_model, data_batches, model_path=saving_path_resnet)
