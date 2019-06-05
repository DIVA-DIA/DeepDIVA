# Utils
import logging
import os

import torch

import numpy as np

# Torch
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_segmentation import load_dataset
from datasets.custom_transform_library import transforms as custom_transforms
from template.setup import _load_mean_std_from_file


def set_up_dataloaders(dataset_folder, batch_size, workers, inmem, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    dataset_folder : string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn
    batch_size : int
        Number of datapoints to process at once
    workers : int
        Number of workers to use for the dataloaders
    inmem : boolean
        Flag : if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.


    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        Dataloaders for train, val and test.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    ###############################################################################################
    # Load the dataset splits as images
    train_ds, val_ds, test_ds = load_dataset(dataset_folder=dataset_folder,
                                    in_memory=inmem,
                                    workers=workers, **kwargs)

    # Loads the analytics csv and extract mean and std
    mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers, kwargs['runner_class'])

    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    # transforms on the image data and ground truth
    # twin crop only during training and validation default is sliding window, which is used during test
    train_ds.transform = val_ds.transform = custom_transforms.Compose([
        custom_transforms.RandomTwinCrop(crop_size=kwargs['crop_size']),
    ])

    # transforms on the image data
    img_transform = transforms.Normalize(mean=mean, std=std)

    # transforms on the gt
    gt_transform = transforms.Compose([
        # transforms the gt image into a one-hot encoded matrix
        custom_transforms.OneHotEncodingDIVAHisDB(class_encodings=train_ds.class_encodings),
        # transforms the one hot encoding to argmax labels -> for the cross-entropy criterion
        custom_transforms.OneHotToPixelLabelling()])

    train_ds.img_transform = val_ds.img_transform = test_ds.img_transform = img_transform
    train_ds.gt_transform = val_ds.gt_transform = test_ds.gt_transform = gt_transform

    # Setup dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             pin_memory=True)

    # the number of workers has to be 1 during testing
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              num_workers=1,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader


def output_to_class_encodings(output, class_encodings, perform_argmax=True):
    """
    This function converts the output prediction matrix to an image like it was provided in the ground truth

    Parameters
    -------
    output : np.array of size [#C x H x W]
        output prediction of the network for a full-size image, where #C is the number of classes
    class_encodings : List
        Contains the range of encoded classes
    perform_argmax : bool
        perform argmax on input data
    Returns
    -------
    numpy array of size [C x H x W] (BGR)
    """

    B = np.argmax(output, axis=0) if perform_argmax else output

    class_to_B = {i: j for i, j in enumerate(class_encodings)}

    masks = [B == old for old in class_to_B.keys()]

    for mask, (old, new) in zip(masks, class_to_B.items()):
        B = np.where(mask, new, B)

    rgb = np.dstack((np.zeros(shape=(B.shape[0], B.shape[1], 2), dtype=np.int8), B))

    return rgb


