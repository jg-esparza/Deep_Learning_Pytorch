import torch
from tqdm import tqdm
from torchmetrics.classification import Dice, JaccardIndex, Precision, Recall

from unet import UNet
from dataset import SegmentationDataset, DataLoader, split_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_dice_score(pred, targets, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).float()
    targets = targets.float()
    intersection = (pred * targets).sum()
    total_sum = pred.sum() + targets.sum() + eps
    dice = (2. * intersection + eps) / total_sum
    return dice.item()


def calculate_iou_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    targets = targets.float()
    intersect = (pred * targets).sum()
    union = pred.sum() + targets.sum() - intersect
    return (intersect/union).item()


def precision_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    intersect = (pred * targets).sum()
    total_pixel_pred = pred.sum()
    precision = intersect/total_pixel_pred
    return precision.item()


def recall_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    intersect = (pred*targets).sum()
    total_pixel_truth = targets.sum()
    recall = intersect/total_pixel_truth
    return recall.item()


def calculate_metrics(outputs, masks, metrics, threshold):
    preds = torch.sigmoid(outputs)  # Convertir a probabilidades
    metrics['dice_scores'].append(calculate_dice_score(preds, masks, threshold))
    metrics['iou_scores'].append(calculate_iou_score(preds, masks, threshold))
    metrics['precision_scores'].append(precision_score(preds, masks, threshold))
    metrics['recall_scores'].append(recall_score(preds, masks, threshold))
    return metrics


def show_result(dice_scores, iou_scores, precisions, recalls):
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    print(f"📌 Test Dice Score: {avg_dice:.4f}")
    print(f"📌 Test IoU Score: {avg_iou:.4f}")
    print(f"📌 Test Precision: {avg_precision:.4f}")
    print(f"📌 Test Recall: {avg_recall:.4f}")


def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    model.to(device)
    metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'precision_scores': [],
        'recall_scores': [],
    }
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            metrics = calculate_metrics(outputs, masks, metrics, threshold)
        show_result(metrics['dice_scores'], metrics['iou_scores'], metrics['precision_scores'], metrics['recall_scores'])


if __name__ == '__main__':
    image_dir = '../datasets/Brain_tumor_segmentation/images'
    mask_dir = '../datasets/Brain_tumor_segmentation/masks'
    dataset = SegmentationDataset(image_dir, mask_dir)
    train_set, val_set, test_set = split_dataset(dataset, train_percent=0.8, val_percent=0.101)

    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    unet = UNet(in_channels=3, out_channels=1)

    unet.load_state_dict(torch.load("./models/unet_14_03.pth", weights_only=True))
    evaluate_model(unet, test_loader, threshold=0.5)
