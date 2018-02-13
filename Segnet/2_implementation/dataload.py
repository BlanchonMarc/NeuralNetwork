import numpy as np
import os
import torch
import torchvision
from torch import autograd

from PIL import Image

from torch.utils.data import Dataset


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def image_path(root, basename, extension):
    return os.path.join(root, basename + extension)


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class DatasetLoader(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'targets')

        self.filenames = [os.path.basename(os.path.splitext(f)[0])
                          for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            target = Image.open(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.filenames)
