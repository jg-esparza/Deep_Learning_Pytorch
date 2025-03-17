import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime

from dataset import SegmentationDataset, split_dataset, DataLoader
from plot import show_graphs
from evaluate import calculate_dice_score, calculate_iou_score, evaluate_model
from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_summary_epoch(num_epochs, epoch, metrics):
    print(f"Epoch {epoch + 1}/{num_epochs} "
          f"- Train Loss: {metrics['train_losses'][-1]:.4f} "
          f"- Val Loss: {metrics['val_losses'][-1]:.4f} "
          f"- Dice Score: {metrics['dice_scores'][-1]:.4f}  "
          f"- Io Score: {metrics['iou_scores'][-1]:.4f}")


def train(model, criterion, optimizer, train_loader, num_epochs, epoch, metrics):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training " + str(epoch + 1) + "/" + str(num_epochs)):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    metrics['train_losses'].append(running_loss / len(train_loader))
    return metrics


def validation(model, criterion, val_loader, num_epochs, epoch, metrics):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating" + str(epoch + 1) + "/" + str(num_epochs)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            pred = torch.sigmoid(outputs)
            dice_score += calculate_dice_score(pred, masks)
            iou_score += calculate_iou_score(pred, masks)
        metrics['val_losses'].append(val_loss / len(val_loader))
        metrics['dice_scores'].append(dice_score / len(val_loader))
        metrics['iou_scores'].append(iou_score / len(val_loader))
    return metrics


def better_validation_loss(metrics, best_val):
    return True if metrics[-1] < best_val else False


def save_model(model, best_model_wts, model_save_path):
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)


def train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    start_time = datetime.datetime.now()
    print(f'Starting training {start_time}')
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'dice_scores': [],
        'iou_scores': []
    }
    best_val = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        metrics = train(model, criterion, optimizer, train_loader, num_epochs, epoch, metrics)
        # Validation phase
        metrics = validation(model, criterion, val_loader, num_epochs, epoch, metrics)
        show_summary_epoch(num_epochs, epoch, metrics)
        if better_validation_loss(metrics['val_losses'], best_val):
            best_val = metrics['val_losses'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    save_model(model, best_model_wts, model_save_path)
    show_graphs(metrics['train_losses'], metrics['val_losses'])
    print(f'Total running time {datetime.datetime.now() - start_time}')


if __name__ == '__main__':
    epochs = 6
    learning_rate = 1e-4
    batch_size = 16
    train_percent, val_percent = 0.8045, 0.115
    # train_percent, val_percent = 0.08045, 0.0115

    image_dir = '../datasets/Brain_tumor_segmentation/images'
    mask_dir = '../datasets/Brain_tumor_segmentation/masks'
    model_save_path = './models/unet_14_03.pth'
    dataset = SegmentationDataset(image_dir, mask_dir)

    train_set, val_set, test_set = split_dataset(dataset, train_percent=train_percent, val_percent=val_percent)

    print(f'Files for training: {len(train_set)}')
    print(f'Files for validation: {len(val_set)}')
    print(f'Files for testing: {len(test_set)}')

    training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testing_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    unet_model = UNet(in_channels=3, out_channels=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(unet_model.parameters(), lr=learning_rate)
    train_loop(unet_model, training_loader, validation_loader, criterion, optimizer, epochs, model_save_path)
    unet_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    evaluate_model(unet_model, testing_loader, threshold=0.5)
