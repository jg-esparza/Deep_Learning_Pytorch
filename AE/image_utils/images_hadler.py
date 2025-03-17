from PIL import Image


def load_images(image_path, format):
    return Image.open(image_path).convert(format)