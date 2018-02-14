'''
Database Loader -> Master Class // Loading Data from Folder
DatabaseTorch -> Sub Class // ``
'''

"""Hyper Class to define a loader of images

This module contains methods to create loaders of images.

The module structure is the following:

- The ``DatabaseLoader`` abstract base class is the main definition of
  the necessary functions in order to properly define an image loader
  from folder
- ``DatabaseTorch`` implements is a derived class that use PyTorch to
  load images from a folder containing a dataset and convert those image
  into torch.Tensor format
"""

from typing import List, Dict
import torch
from torchvision import datasets, models, transforms
import glob
import os


class DatabaseLoader:
    """Abstract Base Class to ensure the optimal quantity of functions""""
    def __init__(self) -> None:
        """Default __init__ to optimize the number of saved arguments"""
        self.root = ''
        self.train_folders = ''
        self.val_folders = ''
        self.ext = ''
        self.sizes = []
        self.output = {}

    def __call__(self) -> Dict[str, torch.Tensor]:
        """Default __call__ returning the converted dataset """
        raise NotImplementedError

    def _to_tensor(self, inds : int, inde : int, ds : dict,
                   st = str) -> torch.Tensor:
       """Default private function to step-wise process data """
        raise NotImplementedError


class DatabaseTorch(DatabaseLoader):
    """Derived Class to process dataset and sub-divide images into proper format

    Attributes
    ----------
    root : str
        The folder containing the dataset subdivided into subfolder.
        Commonly:
        DatasetName/
          |– train/ || training/                (1)
          |– val/ || validate/ || validation/   (2)
          |– test/ || testing/                  (3)
          |– trainannot/ || trainannotation/    (4)
          |– valannot/ || valannotation/        (5)
          |– testannot/ || testannotation/      (6)

    train_folders : List[str]
        The list of subfolders containing the inputs data
        Commonly this List will contain [(1), (2), (3)]

    val_folders : List[str]
        The list of subfolders containing the validation data
        Commonly this List will contain [(4), (5), (6)]

    ext : str
        The string corresponding to the extansion of the dataset
        Commonly "png", "jpg", "tiff" ...
    """
    def __init__(self, root : str, train_folders : List[str],
                 val_folders : List[str], ext : str = 'png') -> None:
        '''Initialization throwing errors for incorrect/incoherent parameters'''
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
                    os.walk(root))[1][x] + '/', "*."+ ext))
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
        """Process the data at call after initialization.

        Parameters
        ---------
        batch_size : int
            The batch_size.

        shuffle : bool
            The shuffling parameter for the data loading.

        num_workers: int
            The number of workers employed in the dataloading.

        Returns
        -------
        output : dict(str, torch.Tensor)
        """

        parent_folder = self.root.split('/')[len(
            self.root.split('/'))-2] + '/'
        newroot = self.root.replace( parent_folder, '')

        image_datasets = {x: datasets.ImageFolder(os.path.join(
            newroot, x),transforms.ToTensor()) for x in [parent_folder]}

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
            inde = inds + conc[1]
            _conc = conc[0].replace('/', '')
            self.output[_conc] = self._to_tensor(inds = inds,
                                                 inde = inde,
                                                 ds = image_datasets,
                                                 st = parent_folder)
            inds = inde
        return self.output

    def _to_tensor(self, inds : int, inde : int, ds : dict,
                   st : str) -> torch.Tensor:
       """Convert a dictionnary of toch.Tensor into a toch.Tensor.

       Sub-divide a dictionnary into a tensor considering sizes predefined.
       
       Parameters
       ---------
       inds : int
           The strating index.

       inds : bool
           The ending index.

       ds: int
           The dictionnary to sub-divide.

       st : str
            The identifier to acces the dictionnary

       Returns
       -------
       output : torch.Tensor
       """
        tmp_storage = []
        for indx in range(inds,inde):
            tmp_storage.append(ds[st][indx][0])
        return torch.stack(tmp_storage)
