"""
Load a dataset of bidimensional points by specifying the folder where its located.
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
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/bd_xor"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    Returns
    -------
    train_ds : data.Dataset
    val_ds : data.Dataset
    test_ds : data.Dataset
        Train, validation and test splits
    """
    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train', 'data.csv')
    val_dir = os.path.join(dataset_folder, 'val', 'data.csv')
    test_dir = os.path.join(dataset_folder, 'test', 'data.csv')

    # Sanity check on the splits folders
    if not os.path.exists(train_dir):
        logging.error("Train data.csv not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.exists(val_dir):
        logging.error("Val data.csv not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.exists(test_dir):
        logging.error("Test data.csv not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # Get the datasets
    train_ds = Bidimensional(train_dir)
    val_ds = Bidimensional(val_dir)
    test_ds = Bidimensional(test_dir)
    return train_ds, val_ds, test_ds


class Bidimensional(data.Dataset):
    """
    This class loads the data.csv file and prepares it as a dataset.
    """

    def __init__(self, path, transform=None, target_transform=None):
        """
        Load the data.csv file and prepare it as a dataset.

        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        """
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
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        point : FloatTensor
        target : int
            label of the point
        """
        x, y, target = self.data[index]

        point = np.array([x, y])
        target = target.astype(np.int64)

        if self.transform is not None:
            # The reshape and scaling are  is absolutely necessary as torch.transform.ToTensor()
            # converts a PIL.Image(RGB) or numpy.ndarray (H x W x C) in the range [0, 255] to a
            # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

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
