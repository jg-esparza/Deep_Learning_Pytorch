import matplotlib.pyplot as plt


def show_sample(image, mask, prediction=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title("Mask")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    if prediction is None:
        plt.imshow(image.permute(1, 2, 0), cmap='gray')
        plt.imshow(mask.squeeze(), cmap="gray", alpha=0.5)
        plt.title("Image + mask")
    else:
        plt.imshow(mask.squeeze(), cmap="gray")
        plt.imshow(prediction.squeeze(), cmap="gray", alpha=0.5)

        plt.title("Prediction")
    plt.axis("off")
    plt.show()


def show_graphs(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
