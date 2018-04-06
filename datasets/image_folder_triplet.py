"""
This file allows to load a dataset of images by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
import random
from multiprocessing import Pool

import cv2
from tqdm import tqdm
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from PIL import Image


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

    # If its requested online, delegate to torchvision.datasets.ImageFolder()
    if not inmem:
        # Get an online dataset for each split
        train_ds = ImageFolderTriplet(train_dir, train=True, num_triplets=num_triplets)
        val_ds = ImageFolderTriplet(val_dir, train=False, num_triplets=num_triplets)
        test_ds = ImageFolderTriplet(test_dir, train=False, num_triplets=num_triplets)
        return train_ds, val_ds, test_ds
    else:
        logging.info('Loading the datasets into memory')
        train_ds = ImageFolderTripletInMem(train_dir, train=True, num_triplets=num_triplets, workers=workers,
                                           model_expected_input_size=model_expected_input_size)
        val_ds = ImageFolderTripletInMem(val_dir, train=False, num_triplets=num_triplets, workers=workers,
                                         model_expected_input_size=model_expected_input_size)
        test_ds = ImageFolderTripletInMem(test_dir, train=False, num_triplets=num_triplets, workers=workers,
                                          model_expected_input_size=model_expected_input_size)
        return train_ds, val_ds, test_ds
    return


class ImageFolderTriplet(data.Dataset):
    """
    This class makes use of torchvision.datasets.ImageFolder() to create an online dataset.
    Afterward all images are sequentially stored in memory for faster use when paired with dataloders.
    It is responsibility of the user ensuring the dataset actually fits in memory.
    """

    def __init__(self, dataset_folder, train=None, num_triplets=None, transform=None, target_transform=None):
        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.transform = transform
        self.target_transform = target_transform

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(dataset_folder)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        self.file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Set expected class attributes
        self.classes = np.unique(self.labels)

        self.train = train

        if self.train:
            self.triplets = self.generate_triplets(self.labels, num_triplets)
        else:
            self.matches = self.generate_matches(self.labels, num_triplets)

    def generate_matches(self, labels, num_triplets):
        """
        matches has format [anchor, positive/negative, match/not_match]; match = 1, not_match=0
        """
        logging.info('Begin generating matches')
        matches = []
        end = int(num_triplets / 2)
        for i in tqdm(range(num_triplets)):
            if i < end:
                c1 = np.random.randint(0, np.max(labels))
                c1_items = np.where(labels == c1)[0]
                a = random.choice(c1_items)
                p = random.choice(c1_items)
                while a == p:
                    p = random.choice(c1_items)
                matches.append([a, p, 1])
            else:
                c1 = np.random.randint(0, np.max(labels))
                c2 = np.random.randint(0, np.max(labels))
                while c1 == c2:
                    c2 = np.random.randint(0, np.max(labels))
                c1_items = np.where(labels == c1)[0]
                c2_items = np.where(labels == c2)[0]
                a = random.choice(c1_items)
                n = random.choice(c2_items)
                matches.append([a, n, 0])
        assert len(matches) == num_triplets
        return matches

    def generate_triplets(self, labels, num_triplets):
        """
        triplets has format [anchor, positive, negative]
        """
        logging.info('Begin generating triplets')
        triplets = []
        for i in tqdm(range(num_triplets)):
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
        return triplets

    def _load_into_mem(self, path):
        return cv2.imread(path)

    def __getitem__(self, index):
        """
        Parameters:
        -----------

        :param index : int
            Index of the sample

        :return: tuple:
            (image, target) where target is index of the target class.
        """
        # TODO: Make it parameterized to use TOPN or FPR type data returns
        if not self.train:
            a, pn, l = self.matches[index]
            img_a = Image.fromarray(self._load_into_mem(self.file_names[a]))
            img_pn = Image.fromarray(self._load_into_mem(self.file_names[pn]))
            if self.transform is not None:
                img_a = self.transform(img_a)
                img_pn = self.transform(img_pn)
            return img_a, img_pn, l

        # if not self.train:
        #     # a, pn, l = self.matches[index]
        #     l = self.labels[index]
        #     img_a = self._load_into_mem(self.file_names[index])
        #     img_a = Image.fromarray(img_a)
        #     if self.transform is not None:
        #         img_a = self.transform(img_a)
        #     return img_a, l

        a, p, n = self.triplets[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
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
            return len(self.matches)


class ImageFolderTripletInMem(ImageFolderTriplet):
    """
    This class makes use of torchvision.datasets.ImageFolder() to create an online dataset.
    Afterward all images are sequentially stored in memory for faster use when paired with dataloders.
    It is responsibility of the user ensuring the dataset actually fits in memory.
    """

    def __init__(self, dataset_folder, train=None, num_triplets=None, transform=None, target_transform=None, workers=1,
                 model_expected_input_size=(224, 224)):

        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.model_expected_input_size = model_expected_input_size

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(dataset_folder)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        self.file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Load all samples
        pool = Pool(workers)
        self.data = pool.map(self._load_into_mem_and_resize, self.file_names)
        pool.close()

        # Set expected class attributes
        self.classes = np.unique(self.labels)

        self.train = train

        if self.train:
            self.triplets = self.generate_triplets(self.labels, num_triplets)
        else:
            self.matches = self.generate_matches(self.labels, int(num_triplets / 10))

    def _load_into_mem_and_resize(self, path):
        """
        Load an image, resize it into the expected size for the model and convert to PIL image.
        :param path:
        :return:
        """
        img = self._load_into_mem(path)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
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

        # TODO: Make it parameterized to use TOPN or FPR type data returns
        if not self.train:
            a, pn, l = self.matches[index]
            img_a = self.data[a]
            img_pn = self.data[pn]
            img_a = Image.fromarray(img_a)
            img_pn = Image.fromarray(img_pn)
            if self.transform is not None:
                img_a = self.transform(img_a)
                img_pn = self.transform(img_pn)
            return img_a, img_pn, l

        # if not self.train:
        #     # a, pn, l = self.matches[index]
        #     l = self.labels[index]
        #     img_a = self.data[index]
        #     img_a = Image.fromarray(img_a)
        #     if self.transform is not None:
        #         img_a = self.transform(img_a)
        #     return img_a, l

        a, p, n = self.triplets[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        img_a = self.data[a]
        img_p = self.data[p]
        img_n = self.data[n]

        img_a = Image.fromarray(img_a)
        img_p = Image.fromarray(img_p)
        img_n = Image.fromarray(img_n)

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
