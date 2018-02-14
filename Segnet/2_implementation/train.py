import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor,
                          torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


def LoadData(data_dir='data', encapsulated_folders=['images', 'targets'],
             train_only=False, train_first=True, crop_size=256,
             nb_classes=21):

    ERROR = 'encapsulated folder should embed two folder names'
    # checking
    assert len(encapsulated_folders) == 2, ERROR

    # DATA AUGMENTATION
    data_transforms = {
        encapsulated_folders[0]: transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        encapsulated_folders[1]: transforms.Compose([
            transforms.CenterCrop(crop_size),
            ToLabel(),
            Relabel(255, nb_classes),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in encapsulated_folders}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4, shuffle=True,
                                                  num_workers=4)
                   for x in encapsulated_folders}

    dataset_sizes = {x: len(image_datasets[x]) for x in encapsulated_folders}

    # validation is embedded or not
    if not train_only:
        train_size = dataset_sizes[encapsulated_folders[0]] / 2
    else:
        train_size = dataset_sizes[encapsulated_folders[0]]

    if train_first:
        indx = 0
    else:
        indx = train_size

    train_inputs_dataset = []
    train_targets_dataset = []

    returning_array = []

    if not train_only:
        val_inputs_dataset = []
        val_targets_dataset = []
        if train_first:
            for i in xrange(train_size):

                train_inputs_dataset.append(
                    image_datasets[encapsulated_folders[0]][i][0])
                val_inputs_dataset.append(
                    image_datasets[encapsulated_folders[0]][i + indx][0])

                train_targets_dataset.append(
                    image_datasets[encapsulated_folders[1]][i][0])
                val_targets_dataset.append(
                    image_datasets[encapsulated_folders[1]][i + indx][0])

            train_inputs_dataset = torch.stack(train_inputs_dataset)
            returning_array.append(train_inputs_dataset)
            val_inputs_dataset = torch.stack(val_inputs_dataset)
            returning_array.append(val_inputs_dataset)
            train_targets_dataset = torch.stack(train_targets_dataset)
            returning_array.append(train_targets_dataset)
            val_targets_dataset = torch.stack(val_targets_dataset)
            returning_array.append(val_targets_dataset)

            returning_array.append(train_size)
            returning_array.append(indx)

            return returning_array
        else:
            for i in xrange(train_size):

                val_inputs_dataset.append(
                    image_datasets[encapsulated_folders[0]][i][0])
                train_inputs_dataset.append(
                    image_datasets[encapsulated_folders[0]][i + train_size][0])

                val_targets_dataset.append(
                    image_datasets[encapsulated_folders[1]][i][0])
                train_targets_dataset.append(
                    image_datasets[encapsulated_folders[1]][i + train_size][0])

            train_inputs_dataset = torch.stack(train_inputs_dataset)
            returning_array.append(train_inputs_dataset)
            val_inputs_dataset = torch.stack(val_inputs_dataset)
            returning_array.append(val_inputs_dataset)
            train_targets_dataset = torch.stack(train_targets_dataset)
            returning_array.append(train_targets_dataset)
            val_targets_dataset = torch.stack(val_targets_dataset)
            returning_array.append(val_targets_dataset)

            returning_array.append(train_size)
            returning_array.append(indx)

            return returning_array
    else:
        for i in xrange(train_size):

            train_inputs_dataset.append(
                image_datasets[encapsulated_folders[0]][i][0])

            train_targets_dataset.append(
                image_datasets[encapsulated_folders[1]][i][0])

        train_inputs_dataset = torch.stack(train_inputs_dataset)
        returning_array.append(train_inputs_dataset)
        train_targets_dataset = torch.stack(train_targets_dataset)
        returning_array.append(train_targets_dataset)

        returning_array.append(train_size)
        returning_array.append(indx)

        return returning_array


# train_only = False
# train_first = True
#
# args = LoadData(train_only=train_only, train_first=train_first)
#
# train_inputs_dataset = args[0]
# val_inputs_dataset = args[1]
# train_targets_dataset = args[2]
# val_targets_dataset = args[3]
# train_size = args[4]
# indx = args[5]
#
# print(train_inputs_dataset)
# print(val_inputs_dataset)


# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in encapsulated_folders}

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in encapsulated_folders}

# print(next(os.walk('data/images/'))[1])

data = [ datasets.ImageFolder(os.path.join('data/images'), transforms.ToTensor()) for x in next(os.walk('data/images/'))[1] if x == 'train']
image_datasets = torch.utils.data.DataLoader(data[0])

print(image_datasets)
