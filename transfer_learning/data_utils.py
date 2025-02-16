
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_dataset(root, transform):
    """Load dataset from directory and transform it."""
    return datasets.ImageFolder(root=root, transform=transform)


def load_batch(train_data, batch_size, shuffle=False, num_workers=0):
    """Generates sample batches."""
    return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def getting_classes_names(dataset):
    """Get class names from dataset."""
    return dataset.classes


def getting_number_of_classes(dataset):
    """Return number of classes in dataset."""
    return len(getting_classes_names(dataset))


def getting_total_batch_sizes(train_loader, test_loader):
    """Get total batch sizes."""
    return {'train': len(train_loader), 'test': len(test_loader)}


def create_dict_dataloaders(train_loader, test_loader):
    """Saves train and test batches into dictionary."""
    return {
        'train': train_loader,
        'test': test_loader
    }


def iterate_dataloader(dataloader):
    """Iterate dataloader."""
    return next(iter(dataloader))


def getting_training_dataset_size(dataset, training_percentage):
    """Get size of training dataset."""
    return int(training_percentage * len(dataset))


def getting_testing_dataset_size(dataset, train_size):
    """Get size of testing dataset."""
    return len(dataset) - train_size


def split_dataset(dataset, training_percentage=0.8):
    """Split dataset into training and testing set."""
    train_size = getting_training_dataset_size(dataset, training_percentage)
    print('Training data: {}'.format(train_size))
    test_size = getting_testing_dataset_size(dataset, train_size)
    print('Testing data: {}'.format(test_size))
    return random_split(dataset, [train_size, test_size])
