# Utils
from __future__ import print_function

import logging
import os

# Torch
import torch
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_triplet import load_dataset
from template.runner.triplet.transforms import MultiCrop
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file


def setup_dataloaders(model_expected_input_size, dataset_folder, n_triplets, batch_size, workers, inmem, multi_crop, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    :param model_expected_input_size: tuple
        Specify the height and width that the model expects.

    :param dataset_folder: string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

    :param n_triplets: int
        Number of triplets to generate for train/val/tes

    :param batch_size: int
        Number of datapoints to process at once

    :param workers: int
        Number of workers to use for the dataloaders

    :param inmem: boolean
        Flag: if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.

    :param multi_crop: int
        Number of crops to use per input image

    :param kwargs: dict
        Any additional arguments.

    :return: dataloader, dataloader, dataloader, int
        Three dataloaders for train, val and test. Number of classes for the model.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    ###############################################################################################
    # Load the dataset splits as images
    train_ds, val_ds, test_ds = load_dataset(dataset_folder=dataset_folder,
                                             in_memory=inmem,
                                             workers=workers,
                                             num_triplets=n_triplets)

    # Loads the analytics csv and extract mean and std
    mean, std = _load_mean_std_from_file(dataset_folder=dataset_folder,
                                         inmem=inmem,
                                         workers=workers)

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    standard_transform = transforms.Compose([
        transforms.Resize(size=model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_ds.transform = standard_transform
    val_ds.transform = standard_transform
    test_ds.transform = standard_transform

    train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       val_ds=val_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)
    return train_loader, val_loader, test_loader
