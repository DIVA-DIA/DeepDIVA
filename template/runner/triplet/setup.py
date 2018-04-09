# Utils
from __future__ import print_function

import os
import logging

import cv2
# Torch
import torch
import torch.nn.init
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_triplet import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file
from template.runner.triplet.transforms import MultiCrop


def setup_dataloaders(model_expected_input_size, dataset_folder, n_triplets, batch_size, workers, inmem, **kwargs):
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
    train_ds, val_ds, test_ds = load_dataset(dataset_folder, inmem, workers, n_triplets, model_expected_input_size)

    # Loads the analytics csv and extract mean and std
    mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers)

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    train_ds.transform = transforms.Compose([
        transforms.CenterCrop(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_ds.transform = transforms.Compose([
        transforms.CenterCrop(size=model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_ds.transform = transforms.Compose([
        transforms.CenterCrop(size=model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_loader, train_loader, val_loader = _dataloaders_from_datasets(batch_size, train_ds, val_ds, test_ds,
                                                                       workers)
    return train_loader, val_loader, train_loader, None
