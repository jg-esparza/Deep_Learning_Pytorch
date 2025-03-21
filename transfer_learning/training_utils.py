import copy
import torch
from data_utils import iterate_dataloader

from data_utils import getting_total_batch_sizes
from plotting import imshow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def copy_best_model_weights(model):
    """Copies model weights from one model to another."""
    return copy.deepcopy(model.state_dict())


def show_current_epoch(epoch, num_epochs):
    """Displays current epoch progress."""
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


def getting_running_loss(loss, inputs):
    """Calculates running loss."""
    return loss.item() * inputs.size(0)


def getting_running_corrects(predicted, labels):
    """Calculates running accuracy."""
    return torch.sum(predicted == labels.data)  # noqa


def getting_epoch_loss(running_loss, total_batch_sizes, phase):
    """Calculates running loss in one epoch."""
    return running_loss / total_batch_sizes[phase]


def getting_epoch_accuracy(running_corrects, total_batch_sizes, phase, batch_size):
    """Calculates running accuracy in one epoch."""
    return running_corrects.double() / (total_batch_sizes[phase] * batch_size)  # noqa


def show_epoch_summary(phase, epoch_loss, epoch_acc):
    """Displays epoch summary."""
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


def show_best_accuracy(best_acc):
    """Displays best accuracy."""
    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))


def is_in_training(phase):
    """Returns whether phase is in training phase."""
    return True if phase == 'train' else False


def is_in_testing(phase):
    """Returns whether phase is in testing phase."""
    return True if phase == 'test' else False


def save_model(model, path):
    """Saves model in the directory path."""
    print('Saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Loads model from the directory path."""
    return model.load_state_dict(torch.load(path))


def train_model(model, criterion, optimizer, scheduler, dataloaders, batch_size, num_epochs=20):
    """Trains model on the given dataset.

    Encapsulates training loop"""
    model = model.to(device)

    best_acc = 0.0
    best_model_wts = copy_best_model_weights(model)

    total_batch_sizes = getting_total_batch_sizes(dataloaders['train'], dataloaders['test'])
    for epoch in range(num_epochs):

        show_current_epoch(epoch, num_epochs)

        for phase in ['train', 'test']:
            if is_in_training(phase):
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if is_in_training(phase):
                        loss.backward()
                        optimizer.step()

                running_loss += getting_running_loss(loss, inputs)
                running_corrects += getting_running_corrects(predicted, labels)

            epoch_loss = getting_epoch_loss(running_loss, total_batch_sizes, phase)
            epoch_acc = getting_epoch_accuracy(running_corrects, total_batch_sizes, phase, batch_size)  # noqa

            show_epoch_summary(phase, epoch_loss, epoch_acc)

            if is_in_testing(phase) and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy_best_model_weights(model)

    show_best_accuracy(best_acc)
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloader):
    """Evaluates model on the given dataset.

    Encapsulates evaluation loop"""
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()   # noqa
        print('Accuracy of the model on the test images: {}%'.format(100 * correct / total))


def show_predictions(model, dataloader, class_names):
    """Displays predictions on given dataset."""
    model.eval()
    with torch.no_grad():
        inputs, labels = iterate_dataloader(dataloader)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(inputs)):
            inp = inputs.data[j]
            imshow(inp, 'predicted:' + class_names[predicted[j]])
