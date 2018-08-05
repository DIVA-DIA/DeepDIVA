"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
from multiprocessing import Pool
import cv2
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from PIL import Image

from util.misc import get_all_files_in_folders_and_subfolders, has_extension


def load_dataset(dataset_folder, in_memory=False, workers=1):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/cifar"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    In each of the three splits (train, val, test) should have different classes in a separate folder
    with the class name. The file name can be arbitrary i.e. it does not have to be 0-* for classes 0
    of MNIST.

    Example:

        train/dog/whatever.png
        train/dog/you.png
        train/dog/like.png

        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932_.png

        train/"class_name"/*.png

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    workers: int
        Number of workers to use for the dataloaders

    Returns
    -------
    train_ds : data.Dataset

    val_ds : data.Dataset

    test_ds : data.Dataset
        Train, validation and test splits
    """
    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # If its requested online, delegate to torchvision.datasets.ImageFolder()
    if not in_memory:
        # Get an online dataset for each split
        train_ds = torchvision.datasets.ImageFolder(train_dir)
        val_ds = torchvision.datasets.ImageFolder(val_dir)
        test_ds = torchvision.datasets.ImageFolder(test_dir)
        return train_ds, val_ds, test_ds
    else:
        # Get an offline (in-memory) dataset for each split
        train_ds = ImageFolderInMemory(train_dir, workers)
        val_ds = ImageFolderInMemory(val_dir, workers)
        test_ds = ImageFolderInMemory(test_dir, workers)
        return train_ds, val_ds, test_ds


class ImageFolderInMemory(data.Dataset):
    """
    This class loads the data provided and stores it entirely in memory as a dataset.

    It makes use of torchvision.datasets.ImageFolder() to create a dataset. Afterward all images are
    sequentially stored in memory for faster use when paired with dataloders. It is responsibility of
    the user ensuring that the dataset actually fits in memory.
    """

    def __init__(self, path, transform=None, target_transform=None, workers=1):
        """
        Load the data in memory and prepares it as a dataset.

        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        workers: int
            Number of workers to use for the dataloaders
        """
        self.dataset_folder = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(path)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Load all samples
        pool = Pool(workers)
        self.data = pool.map(cv2.imread, file_names)
        pool.close()

        # Set expected class attributes
        self.classes = np.unique(self.labels)

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        img : FloatTensor
        target : int
            label of the image
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


class ImageFolderApply(data.Dataset):
    """
    TODO fill me
    """

    def __init__(self, path, transform=None, target_transform=None, classify=False):
        """
        TODO fill me

        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        """
        self.dataset_folder = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        if classify is True:
            # Get an online dataset
            dataset = torchvision.datasets.ImageFolder(path)

            # Extract the actual file names and labels as entries
            self.file_names = np.asarray([item[0] for item in dataset.imgs])
            self.labels = np.asarray([item[1] for item in dataset.imgs])
        else:
            # Get all files in the folder that are images
            self.file_names = self._get_filenames(self.dataset_folder)

            # Extract the label for each file (assuming standard format of root_folder/class_folder/img.jpg)
            self.labels = [item.split('/')[-2] for item in self.file_names]

        # Set expected class attributes
        self.classes = np.unique(self.labels)

    def _get_filenames(self, path):
        file_names = []
        for item in get_all_files_in_folders_and_subfolders(path):
            if has_extension(item, ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                file_names.append(item)
        return file_names

    def __getitem__(self, index):
        """
        Retrieve a sample by index and provides its filename as well

        Parameters
        ----------
        index : int

        Returns
        -------
        img : FloatTensor
        target : int
            label of the image
        filename : string
        """

        # Weird way to open things due to issue https://github.com/python-pillow/Pillow/issues/835
        with open(self.file_names[index], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        target, filename = self.labels[index], self.file_names[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, filename

    def __len__(self):
        return len(self.file_names)
