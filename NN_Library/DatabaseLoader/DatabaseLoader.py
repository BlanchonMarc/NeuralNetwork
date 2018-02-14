'''
Database Loader -> Master Class // Loading Data from Folder
DatabaseTorch -> Sub Class // ``
'''

from typing import List
import torch
import os


class DatabaseLoader:
    def __init__(self, root=str, train_folders=List[str], val_folder=List[str]):
        self.root = root
        self.train_folders = train_folders
        self.val_folder = val_folder


class DatabaseTorch(DatabaseLoader):
    def __init__(self, root=str, train_folders=List[str], val_folder=List[str]):
        DatabaseLoader.__init__(self)
        if os.path.isdir(root):
            self.root = root
        else:
            raise NameError('Incorrect path')

        folders_safety = [os.path.isdir(root + folder) for folder in train_folders]

        if False in folders_safety:
            raise NameError('Incorrect path for training folders selection')
        else:
            self.train_folders = train_folders

        folders_safety = [os.path.isdir(root + folder) for folder in val_folder]

        if False in folders_safety:
            raise NameError('Incorrect path for training folders selection')
        else:
            self.val_folder = val_folder


# Testing procedure
DatabaseTorch('../../Datasets/', ['CamVid/'] ,['test/'])

print(next(os.walk('../../Datasets/'))[1])
