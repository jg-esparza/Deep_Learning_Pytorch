import os


def list_directory(path):
    return os.listdir(path)


def join_directories(image_directory, image_names):
    return os.path.join(image_directory, image_names)