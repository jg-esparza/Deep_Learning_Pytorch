import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from datetime import datetime

from models import CustomCNN, ResNet
from data_utils import (mean, std, load_dataset, load_batch, getting_classes_names, getting_number_of_classes,
                        create_dict_dataloaders)
from training_utils import train_model, evaluate_model, save_model, load_model


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def getting_dataloaders(train_data, test_data, batch_size): # noqa
    """Return batches of training and testing datasets in a dictionary.

    Load training and testing data from the provided directories,
    Use dataloaders to get batches of training and testing data.
    """
    train_loader = load_batch(train_data, batch_size=batch_size, shuffle=True)
    test_loader = load_batch(test_data, batch_size=batch_size, shuffle=False)
    return create_dict_dataloaders(train_loader, test_loader)


def training_classifier(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,  # noqa
                        num_epochs, batch_size, path = None):  # noqa
    """Training a classifier with predefined criterion, optimizer and learning rate scheduler.

    Saves the model if provides a directory.
    """
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                        batch_size, num_epochs=num_epochs)

    if path is not None:
        save_model(model, path)
    return model


def evaluating_classifier(model, dataloaders, model_path=None): # noqa
    """Evaluates a classifier using testing data.

    Loads a trained model if provides a directory.
    """
    if model_path is not None:
        load_model(model, model_path)
    evaluate_model(model, dataloaders['test'])


def run_classifier(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs, # noqa
                   batch_size, saving_path=None):   # noqa
    """Train and evaluate a classifier."""
    start_time = datetime.now()
    model = training_classifier(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                                num_epochs, batch_size, path=saving_path)
    end_time = datetime.now()
    print('Total training time: {}'.format(end_time - start_time))
    evaluating_classifier(model, dataloaders, saving_path)


if __name__ == '__main__':
    epochs = 5
    batch_size = 16
    train_dir = '../datasets/cifar10/train'
    test_dir = '../datasets/cifar10/test'
    # train_data = load_dataset(root=train_dir, transform=transforms)
    # test_data = load_dataset(root=test_dir, transform=transforms)
    train_data = torchvision.datasets.CIFAR10(
        root='../datasets/cifar10/train',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_data = torchvision.datasets.CIFAR10(
        root='../datasets/cifar10/test',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    dataloaders = getting_dataloaders(train_data, test_data, batch_size)

    num_classes = getting_number_of_classes(train_data)
    custom_cnn = CustomCNN(num_classes=num_classes)
    # print('Custom CNN: {}'.format(custom_cnn))
    # define a loss function
    criterion = nn.CrossEntropyLoss()
    # define optimizer
    optimizer_ft = optim.SGD(custom_cnn.parameters(), lr=0.001, momentum=0.9)
    # define learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    run_classifier(custom_cnn, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, epochs,
                   batch_size, '../models/Custom_CNN.pth')
    # evaluating_classifier(custom_cnn, data_batches, model_path='../models/Custom_CNN.pth')

    # Resnet18
    resnet18_model = ResNet(num_classes=num_classes, pretrained=True)
    # define optimizer
    optimizer_ft = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9)
    # define learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    # print('Resnet18_freezing top layers: {}'.format(resnet18_model))
    saving_path_resnet = '../models/Resnet18.pth'
    print(saving_path_resnet)
    run_classifier(resnet18_model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, epochs,
                   batch_size, saving_path=saving_path_resnet)
    evaluating_classifier(resnet18_model, dataloaders, model_path=saving_path_resnet)

    # Resnet18 freeze top layers
    resnet18_model = ResNet(num_classes=num_classes, pretrained=True, freeze_layers=4)
    # define optimizer
    optimizer_ft = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9)
    # define learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    # print('Resnet18_freezing top layers: {}'.format(resnet18_model))
    saving_path_resnet = '../models/Resnet18_freeze_top_layers.pth'
    print(saving_path_resnet)
    run_classifier(resnet18_model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, epochs,
                   batch_size, saving_path=saving_path_resnet)
    # evaluating_classifier(resnet18_model, dataloaders, model_path=saving_path_resnet)
