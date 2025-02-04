import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from training_cnns import training_loop, show_accuracy, show_confusion_matrix
from plotting import plot_loss_per_iteration


class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, k)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


def load_dataset_fashion_mnist(directory, train):
    """Load dataset for fashion mnist dataset"""
    return torchvision.datasets.FashionMNIST(
        root=directory,
        train=train,
        transform=transforms.ToTensor(),
        download=True)


def getting_batches(dataset, defined_batch_size, shuffle):
    """Automatically generates batches in the training loop and
    takes care of shuffling"""
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=defined_batch_size,
                                       shuffle=shuffle)


def number_of_classes_in(dataset):
    """Number of classes in the dataset"""
    return len(set(dataset.targets.numpy()))


def train_cnn(model, train_dataset, test_dataset, batch_size, epochs,
              criterion, optimizer):

    train_loader = getting_batches(train_dataset, batch_size, True)

    test_loader = getting_batches(test_dataset, batch_size, False)

    train_losses, test_losses = training_loop(model, criterion, optimizer, train_loader,
                                              test_loader, epochs=epochs)

    plot_loss_per_iteration(train_losses, test_losses)
    show_accuracy(model, train_loader, test_loader)
    show_confusion_matrix(model, test_loader, test_dataset)


if __name__ == '__main__':

    training_dataset = load_dataset_fashion_mnist('../datasets/fashionmnist/train',
                                                  train=True)

    testing_dataset = load_dataset_fashion_mnist('../datasets/fashionmnist/test',
                                                 train=False)   # noqa
    K = number_of_classes_in(training_dataset)
    print("number of classes:", K)

    cnn_model = CNN(K)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters())

    selected_batch_size = 128
    num_epochs = 5
    train_cnn(cnn_model, training_dataset, testing_dataset, selected_batch_size, num_epochs,
              criterion, optimizer)
