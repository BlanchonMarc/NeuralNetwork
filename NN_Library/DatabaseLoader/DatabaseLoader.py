'''
Database Loader -> Master Class // Loading Data from Folder
DatabaseTorch -> Sub Class // ``
'''

from typing import List, Dict
import torch
from torchvision import datasets, models, transforms
import glob
import os


class DatabaseLoader:
    def __init__(self) -> None:
        self.root = ''
        self.train_folders = ''
        self.val_folders = ''
        self.ext = ''
        self.sizes = []
        self.output = {}

    def __call__(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _to_tensor(self, inds : int, inde : int, ds : dict) -> torch.Tensor:
        raise NotImplementedError


class DatabaseTorch(DatabaseLoader):
    """
    Process the specified folder in order to return dict[str,torch.Tensors]
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

        self.sizes = [len(glob.glob1(root + next(
                    os.walk(root))[1][x] + '/', "*.png"))
                    for x in range(len(next(os.walk(root))[1]))]

        if not len(next(os.walk(root))[1]) == len(
            train_folders + val_folders):
            raise AttributeError('#folder in ' + root + ' = ' + str(len(
                next(os.walk(root))[1])) + ' while #folder as input = ' + str(
                    len(self.train_folders + self.val_folders)))

        self.output = {}


    def __call__(self, batch_size : int = 1,
                 shuffle : bool = True,
                 num_workers : int = 4) -> Dict[str, torch.Tensor]:

        parent_folder = self.root.split('/')[len(
            self.root.split('/'))-2] + '/'
        newroot = self.root.replace( parent_folder, '')

        image_datasets = {x: datasets.ImageFolder(os.path.join(newroot, x))
                          for x in [parent_folder]}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=num_workers)
                       for x in [parent_folder]}

        dataset_sizes = [len(image_datasets[x]) for x in [parent_folder]]

        if not sum(self.sizes) == dataset_sizes[0]:
            raise NotImplementedError

        temporary_folders = sorted(self.train_folders + self.val_folders)

        inds = 0
        for key, conc in enumerate(zip(temporary_folders,self.sizes)):
            inde = conc[1]
            _conc = conc[0].replace('/', '')
            self.output[_conc] = self._to_tensor(inds = inds, inde = inde, ds = image_datasets)
            inds = inde

            print('key, size ' + str(conc[1]))

        print(sorted(temporary_folders))
        print(self.sizes)
        print(sum(self.sizes))
        print(dataset_sizes)

    def _to_tensor(self, inds : int, inde : int, ds : dict) -> torch.Tensor:
        pass



# Testing procedure
root_dataset = '../../Datasets/CamVid/'
inputs = ['train/', 'val/', 'test/']
checkings = ['trainannot/', 'valannot/', 'testannot/']
Db = DatabaseTorch(root=root_dataset, train_folders=inputs, val_folders=checkings)

Db(batch_size = 1, shuffle = True, num_workers = 4)
