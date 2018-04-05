"""
Bidimensional dataset
"""

# Utils
import logging
import os
import os.path
import sys

import numpy as np
import pandas
# Torch related stuff
import torch.utils.data as data


def load_dataset(dataset_folder):
    """
    Parameters
    ----------

    :param dataset_folder: string (path)
        Specifies where the dataset is located on the file System

    :return train_ds, val_da, test_da: data.Dataset
        Return a torch dataset for each split

    Structure of the dataset expected
    ---------------------------------

    Split folders:

        'args.dataset_folder' has to point to the three folder train/val/test.
        Example:

        ~/../../data/pc_diagonal

        where the dataset_folder contains the splits sub-folders as follow:

        args.dataset_folder/train
        args.dataset_folder/val
        args.dataset_folder/test
        """

    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train', 'data.csv')
    val_dir = os.path.join(dataset_folder, 'val', 'data.csv')
    test_dir = os.path.join(dataset_folder, 'test', 'data.csv')

    # Sanity check on the splits folders
    if not os.path.exists(train_dir):
        logging.error("Train data.csv not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.exists(val_dir):
        logging.error("Val data.csv not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.exists(test_dir):
        logging.error("Test data.csv not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # Get the datasets
    train_ds = Bidimensional(train_dir)
    val_ds = Bidimensional(val_dir)
    test_ds = Bidimensional(test_dir)
    return train_ds, val_ds, test_ds


class Bidimensional(data.Dataset):
    """
    This class loads the data.csv file and stores it as a dataset.
    """

    def __init__(self, path, transform=None, target_transform=None):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # Read data from the csv file
        self.data = pandas.read_csv(self.path).as_matrix()

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(self.data)

        self.min_coords = np.min(self.data[:, 0]), np.min(self.data[:, 1])
        self.max_coords = np.max(self.data[:, 0]), np.max(self.data[:, 1])

        # Set expected class attributes
        self.classes = np.unique(np.unique(self.data[:, 2]))
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        """
        Parameters:
        -----------

        :param index : int
            Index of the sample

        :return: tuple:
            (point, target) where target is index of the target class.
        """

        x, y, target = self.data[index]

        point = np.array([x, y])
        target = target.astype(np.int64)

        if self.transform is not None:
            """    
            The reshape and scaling are  is absolutely necessary as torch.transform.ToTensor() converts a PIL.Image(RGB)
            or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range 
            [0.0, 1.0].
            """
            # Bring from domain range into [0;255]
            point = np.divide((point - self.min_coords), np.subtract(self.max_coords, self.min_coords)) * 255
            # Reshape into (H x W x C)
            point = point.reshape(1, 1, 2)
            # Apply transforms
            point = self.transform(point)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return point, target

    def __len__(self):
        return len(self.data)
