'''
Database Loader -> Master Class // Loading Data from Folder
DatabaseTorch -> Sub Class // ``
'''

from typing import List
import torch
import glob
import os


class DatabaseLoader:
    def __init__(self) -> None:
        self.root = ''
        self.train_folders = ''
        self.val_folders = ''
        self.ext = ''
        self.sizes = []

    def to_Tensor(self, selection : str) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


class DatabaseTorch(DatabaseLoader):
    """
    Process the specified folder in order to return torch.Tensors
    """
    def __init__(self, root : str, train_folders : List[str],
                 val_folders : List[str], ext : str = '*png') -> None:
        '''
        Safety check for the paths
        '''
        super().__init__()
        if os.path.isdir(root):
            self.root = root
        else:
            raise AttributeError(
                'Incorrect path')

        folders_safety = [os.path.isdir(root + folder)
                         for folder in train_folders]

        if False in folders_safety:
            raise AttributeError(
                'Incorrect path for training folders selection')
        else:
            self.train_folders = train_folders

        folders_safety = [os.path.isdir(root + folder)
                         for folder in val_folders]

        if False in folders_safety:
            raise AttributeError(
                'Incorrect path for training folders selection')
        else:
            self.val_folders = val_folders

        #print(len(glob.glob1()[1], "*.png")))
        self.sizes = [len(glob.glob1(root + next(
                    os.walk(root))[1][x] + '/', "*.png"))
                    for x in range(len(next(os.walk(root))[1]))]

    def __call__(self) -> List[torch.Tensor]:

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in encapsulated_folders}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=4, shuffle=True,
                                                      num_workers=4)
                       for x in encapsulated_folders}

        dataset_sizes = {x: len(image_datasets[x]) for x in encapsulated_folders}

    def to_Tensor(self, selection : str) -> torch.Tensor:
        pass



# Testing procedure
root_dataset = '../../Datasets/Camvid/'
inputs = ['train/', 'val/', 'test/']
checkings = ['trainannot/', 'valannot/', 'testannot/']
Db = DatabaseTorch(root=root_dataset, train_folders=inputs, val_folders=checkings)

Db()
# print(sorted(inputs))
#print(next(os.walk(root_dataset))[1][1])
# tifCounter = len(glob.glob1(root_dataset + 'train/', "*.png"))
# print(tifCounter)
