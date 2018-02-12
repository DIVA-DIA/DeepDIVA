"""
This file allows to load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys

import cv2
import numpy as np
# Torch related stuff
import torch.utils.data as data
import torchvision
from PIL import Image


def load_dataset(dataset_folder, online):
    """
    Parameters
    ----------

    :param dataset_folder: string (path)
        Specifies where the dataset is located on the file System

    :param online: boolean
        Flag: if True, the dataset is loaded in an online fashion i.e. only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    :return train_ds, val_da, test_da: data.Dataset
        Return a torch dataset for each split

    Structure of the dataset expected
    ---------------------------------

    Split folders:

        'args.dataset_folder' has to point to the three folder train/val/test.
        Example:

        ~/../../data/svhn

        where the dataset_folder contains the splits sub-folders as follow:

        args.dataset_folder/train
        args.dataset_folder/val
        args.dataset_folder/test

    Classes folders

        In each of the three splits (train,val,test) should have different classes in a separate folder with the class
        name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).

        Example:

        train/dog/whatever.png
        train/dog/you.png
        train/dog/like.png

        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932_.png
    """

    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # If its requested online, delegate to torchvision.datasets.ImageFolder()
    if online:
        # Get an online dataset for each split
        train_ds = torchvision.datasets.ImageFolder(train_dir)
        val_ds = torchvision.datasets.ImageFolder(val_dir)
        test_ds = torchvision.datasets.ImageFolder(test_dir)
        return train_ds, val_ds, test_ds
    else:
        # Get an offline (in-memory) dataset for each split
        train_ds = ImageFolderInMemory(train_dir)
        val_ds = ImageFolderInMemory(val_dir)
        test_ds = ImageFolderInMemory(test_dir)
        return train_ds, val_ds, test_ds


class ImageFolderInMemory(data.Dataset):
    """
    This class makes use of torchvision.datasets.ImageFolder() to create an online dataset.
    Afterward all images are sequentially stored in memory for faster use when paired with dataloders.
    It is responsibility of the user ensuring the dataset actually fits in memory.
    """

    def __init__(self, dataset_folder, transform=None, target_transform=None, ):
        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.transform = transform
        self.target_transform = target_transform

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(dataset_folder)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Load all samples
        self.data = np.zeros([file_names.size] + list(cv2.imread(file_names[0]).shape), dtype='uint8')
        for i, sample in enumerate(file_names):
            self.data[i] = cv2.imread(sample)

        # Set expected class attributes
        self.classes = np.unique(self.labels)



    def __getitem__(self, index):
        """
        Parameters:
        -----------

        :param index : int
            Index of the sample

        :return: tuple:
            (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
