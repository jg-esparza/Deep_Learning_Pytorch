from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from directory_utils.directory_handler import list_directory, join_directories
from image_utils.images_hadler import load_images


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
             transforms.Resize((256, 256)),
             transforms.ToTensor()])

        self.image_filenames = list_directory(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = join_directories(self.image_dir, self.image_filenames[idx])
        mask_path = join_directories(self.mask_dir, self.image_filenames[idx])

        image = load_images(img_path, "RGB")
        mask = load_images(mask_path, "L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask


def split_dataset(dataset, train_percent=0.8, val_percent=0.1):
    train_size = int(train_percent * len(dataset))
    val_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])
