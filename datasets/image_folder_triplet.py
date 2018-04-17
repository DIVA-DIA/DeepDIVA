"""
This file allows to load a dataset of images by specifying the folder where its located.
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


def load_dataset(dataset_folder, inmem=False, workers=1, num_triplets=None, model_expected_input_size=None, **kwargs):
    """
    Parameters
    ----------

    :param dataset_folder: string (path)
        Specifies where the dataset is located on the file System

    :param inmem: boolean
        Flag: if False, the dataset is loaded in an online fashion i.e. only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    :param workers: int
        Number of workers to use for the dataloaders

    :param num_triplets: int
        Number of triplets to generate from the data

    :param model_expected_input_size: tuple
        Specify the height and width that the model expects.


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

    train_ds = ImageFolderTriplet(train_dir, train=True, num_triplets=num_triplets, workers=workers, inmem=inmem)
    val_ds = ImageFolderTriplet(val_dir, train=False, num_triplets=num_triplets, workers=workers, inmem=inmem)
    test_ds = ImageFolderTriplet(test_dir, train=False, num_triplets=num_triplets, workers=workers, inmem=inmem)
    return train_ds, val_ds, test_ds


class ImageFolderTriplet(data.Dataset):

    def __init__(self, dataset_folder, train=None, num_triplets=None, transform=None, target_transform=None, inmem=None, workers=None):
        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.num_triplets = num_triplets
        self.inmem = inmem

        dataset = torchvision.datasets.ImageFolder(dataset_folder)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each batch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        self.file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Set expected class attributes
        self.classes = np.unique(self.labels)

        if self.train:
            self.triplets = self.generate_triplets()

        if self.inmem:
            # Load all samples
            with Pool(workers) as pool:
                self.data = pool.map(self._load_into_mem_and_resize, self.file_names)

    def generate_triplets(self):
        """
        triplets have format [anchor, positive, negative]
        """
        labels = self.labels
        num_triplets = self.num_triplets
        logging.info('Begin generating triplets')
        triplets = []
        for i in trange(num_triplets, leave=False):
            c1 = np.random.randint(0, np.max(labels))
            c2 = np.random.randint(0, np.max(labels))
            while c1 == c2:
                c2 = np.random.randint(0, np.max(labels))

            c1_items = np.where(labels == c1)[0]
            a = random.choice(c1_items)
            p = random.choice(c1_items)
            while a == p:
                p = random.choice(c1_items)

            c2_items = np.where(labels == c2)[0]
            n = random.choice(c2_items)
            triplets.append([a, p, n])
        logging.info('Finished generating {} triplets'.format(num_triplets))
        return triplets

    def _load_into_mem(self, path):
        return cv2.imread(path)

    def _load_into_mem_and_resize(self, path):
        """
        Load an image, resize it into the expected size for the model and convert to PIL image.
        :param path:
        :return:
        """
        img = self._load_into_mem(path)
        return img

    def __getitem__(self, index):
        """
        Parameters:
        -----------

        :param index : int
            Index of the sample

        :return: tuple:
            (image, target) where target is index of the target class.
        """
        if not self.train:
            # a, pn, l = self.matches[index]
            l = self.labels[index]
            if self.inmem:
                img_a = self.data[index]
            else:
                img_a = self._load_into_mem(self.file_names[index])
            img_a = Image.fromarray(img_a)
            if self.transform is not None:
                img_a = self.transform(img_a)
            return img_a, l

        a, p, n = self.triplets[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        if self.inmem:
            img_a = Image.fromarray(self.data[a])
            img_p = Image.fromarray(self.data[p])
            img_n = Image.fromarray(self.data[n])
        else:
            img_a = Image.fromarray(self._load_into_mem(self.file_names[a]))
            img_p = Image.fromarray(self._load_into_mem(self.file_names[p]))
            img_n = Image.fromarray(self._load_into_mem(self.file_names[n]))

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
