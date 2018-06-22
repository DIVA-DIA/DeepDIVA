"""
Load a dataset of images by specifying the folder where its located and prepares it for triplet
similarity matching training.
"""

# Utils
import logging
import os
import random
import sys
from multiprocessing import Pool
import cv2
import numpy as np
import torch.utils.data as data

# Torch related stuff
import torchvision
from PIL import Image
from tqdm import trange


def load_dataset(dataset_folder, num_triplets=None, in_memory=False, workers=1):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test.

    The dataset is expected to be in the same structure as described in image_folder_dataset.load_dataset()

    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System
    num_triplets : int
        Number of triplets [a, p, n] to generate on dataset creation
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
        logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    train_ds = ImageFolderTriplet(train_dir, train=True, num_triplets=num_triplets,
                                  workers=workers,in_memory=in_memory)
    val_ds = ImageFolderTriplet(val_dir, train=False, num_triplets=num_triplets,
                                workers=workers, in_memory=in_memory)
    test_ds = ImageFolderTriplet(test_dir, train=False, num_triplets=num_triplets,
                                 workers=workers, in_memory=in_memory)
    return train_ds, val_ds, test_ds


class ImageFolderTriplet(data.Dataset):
    """
    This class loads the data provided and stores it entirely in memory as a dataset.
    Additionally, triplets will be generated in the format of [a, p, n] and their file names stored
    in memory.
    """

    def __init__(self, path, train=None, num_triplets=None, in_memory=None,
                 transform=None, target_transform=None, workers=None):
        """
        Load the data in memory and prepares it as a dataset.

        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        train : bool
            Denotes whether this dataset will be used for training. Its very important as for
            validation and test there are no triplet but pairs to evaluate similarity matching.
        num_triplets : int
            Number of triplets [a, p, n] to generate on dataset creation
        in_memory : boolean
            Load the whole dataset in memory. If False, only file names are stored and images are
            loaded on demand. This is slower than storing everything in memory.
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        workers: int
            Number of workers to use for the dataloaders
        """
        self.dataset_folder = os.path.expanduser(path)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.num_triplets = num_triplets
        self.in_memory = in_memory

        dataset = torchvision.datasets.ImageFolder(path)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each batch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        self.file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Set expected class attributes
        self.classes = np.unique(self.labels)

        if self.train:
            self.triplets = self.generate_triplets()

        if self.in_memory:
            # Load all samples
            with Pool(workers) as pool:
                self.data = pool.map(self.cv2.imread, self.file_names)

    def generate_triplets(self):
        """
        Generate triplets for training. Triplets have format [anchor, positive, negative]
        """
        logging.info('Begin generating triplets')
        triplets = []
        for _ in trange(self.num_triplets, leave=False):
            # Select two different classes, c1 and c2
            c1 = np.random.randint(0, np.max(self.labels))
            c2 = np.random.randint(0, np.max(self.labels))
            while c1 == c2:
                c2 = np.random.randint(0, np.max(self.labels))

            # Select two different object of class c1, a and p
            c1_items = np.where(self.labels == c1)[0]
            a = random.choice(c1_items)
            p = random.choice(c1_items)
            while a == p:
                p = random.choice(c1_items)

            # Select an item from class c2, n
            c2_items = np.where(self.labels == c2)[0]
            n = random.choice(c2_items)

            # Add the triplet to the list as we now have a,p,n
            triplets.append([a, p, n])
        logging.info('Finished generating {} triplets'.format(self.num_triplets))
        return triplets

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        img_a : FloatTensor
            Anchor image
        img_p : FloatTensor
            Positive image (same class of anchor)
        img_n : FloatTensor
            Negative image (different class of anchor)
        """
        if not self.train:
            # a, pn, l = self.matches[index]
            l = self.labels[index]
            if self.in_memory:
                img_a = Image.fromarray(self.data[index])
            else:
                img_a = Image.fromarray(cv2.imread(self.file_names[index]))

            if self.transform is not None:
                img_a = self.transform(img_a)
            return img_a, l

        a, p, n = self.triplets[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        if self.in_memory:
            img_a = Image.fromarray(self.data[a])
            img_p = Image.fromarray(self.data[p])
            img_n = Image.fromarray(self.data[n])
        else:
            img_a = Image.fromarray(cv2.imread(self.file_names[a]))
            img_p = Image.fromarray(cv2.imread(self.file_names[p]))
            img_n = Image.fromarray(cv2.imread(self.file_names[n]))

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return len(self.triplets)
        else:
            return len(self.file_names)
