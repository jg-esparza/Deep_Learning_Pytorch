import numpy as np
import torch
from datetime import datetime

from sklearn.metrics import confusion_matrix

from plotting import plot_confusion_matrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def getting_confusion_matrix(model, test_dataset, test_loader):
    """Evaluate the model on the test dataset"""
    y_test = test_dataset.targets.numpy()
    p_test = np.array([])
    for inputs, targets in test_loader:
        _, predictions = evaluate_model(model, inputs, targets)
        p_test = np.concatenate((p_test, predictions.cpu().numpy()))
        return y_test, p_test


def show_confusion_matrix(model, test_loader, test_dataset):
    """Print and plots the confusion matrix."""
    y_test, p_test = getting_confusion_matrix(model, test_dataset, test_loader)
    conf_matrix = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(conf_matrix, list(range(10)))


def evaluate_model(model, inputs, targets):
    """Evaluate the model with the given inputs and targets"""
    # move data to GPU
    inputs, targets = inputs.to(device), targets.to(device)
    # Forward pass
    outputs = model(inputs)
    # Get prediction
    # torch.max returns both max and argmax
    return torch.max(outputs, 1)


def update_counts(predictions, targets):
    return (predictions == targets).sum().item()    # noqa


def show_accuracy(model, train_loader, test_loader):
    """Calculate the accuracy of the model"""
    model.eval()
    n_correct_training = 0.
    n_total_training = 0.
    for inputs, targets in train_loader:
        _, predictions = evaluate_model(model, inputs, targets)
        n_correct_training += update_counts(predictions, targets)
        n_total_training += targets.shape[0]
    train_acc = n_correct_training / n_total_training

    n_correct_testing = 0.
    n_total_testing = 0.
    for inputs, targets in test_loader:
        _, predictions = evaluate_model(model, inputs, targets)
        n_correct_testing += update_counts(predictions, targets)
        n_total_testing += targets.shape[0]
    test_acc = n_correct_testing / n_total_testing
    print(f"Train acc: {train_acc:.6f}, Test acc: {test_acc:.6f}")


def training_loop(model, criterion, optimizer, train_loader, test_loader, epochs):
    """A function to encapsulate the training loop"""
    model.to(device)
    print(model)
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Duration: {dt}')
    return train_losses, test_losses
