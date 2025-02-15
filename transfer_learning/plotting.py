import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from data_utils import mean, std


def make_grid_for_plotting(inputs):
    """Creates a grid to show the samples."""
    return make_grid(inputs)


def imshow(inputs, title):
    """Shows a sample image."""
    inp = make_grid_for_plotting(inputs)
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(12, 6))
    plt.imshow(inp)
    plt.title(title)
    plt.pause(5)
