# Utils
import logging
import os
import pandas as pd


# TODO: from __future__ import print_function
import torch

import numpy as np

# Torch
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_segmentation import load_dataset
from datasets.custom_transform_library import transforms as custom_transforms
from template.setup import _load_mean_std_from_file


def set_up_dataloaders(model_expected_input_size, dataset_folder, batch_size, workers, inmem, **kwargs):
    # TODO: refactor into the image_folder_segmentation.py
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    model_expected_input_size : tuple
        Specify the height and width that the model expects.
    dataset_folder : string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn
    n_triplets : int
        Number of triplets to generate for train/val/tes
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
    # gt_transform = custom_transforms.OneHotEncoding(class_encodings=train_ds.class_encodings)
    # TODO: make the argmax a transform (as not all criterion will want this)
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


def one_hot_to_np_rgb(matrix, class_encodings):
    """
    This function converts the one-hot encoded matrix to an image like it was provided in the ground truth

    Parameters
    -------
    np array of size [#C x H x W]
        sparse one-hot encoded matrix, where #C is the number of classes
    Returns
    -------
    numpy array of size [C x H x W] (BGR)
    """
    B = np.argmax(matrix, axis=0)
    class_to_B = {i: j for i, j in enumerate(class_encodings)}

    masks = [B == old for old in class_to_B.keys()]

    for mask, (old, new) in zip(masks, class_to_B.items()):
        B = np.where(mask, new, B)

    rgb = np.dstack((np.zeros(shape=(B.shape[0], B.shape[1], 2), dtype=np.int8), B))

    return rgb


def one_hot_to_full_output(one_hot, coordinates, combined_one_hot, output_dim):
    """
    This function combines the one-hot matrix of all the patches in one image to one large output matrix. Overlapping
    values are averaged.

    Parameters
    ----------
    output_dims: tuples [Htot x Wtot]
        dimension of the large image
    one_hot: numpy matrix of size [batch size x #C x H x W]
        a patch from the larger image
    coordinates: tuple
        top left coordinates of the patch within the larger image for all patches in a batch
    combined_one_hot: numpy matrix of size [#C x Htot x Wtot]
        one hot encoding of the full image
    Returns
    -------
    combined_one_hot: numpy matrix [#C x Htot x Wtot]
    """
    if len(combined_one_hot) == 0:
        combined_one_hot = np.zeros((one_hot.shape[0], *output_dim))

    x1, y1 = coordinates
    x2, y2 = (min(x1 + one_hot.shape[1], output_dim[0]), min(y1 + one_hot.shape[2], output_dim[1]))
    zero_mask = combined_one_hot[:, x1:x2, y1:y2] == 0
    # if still zero in combined_one_hot just insert value from crop, if there is a value average
    combined_one_hot[:, x1:x2, y1:y2] = np.where(zero_mask, one_hot[:, :zero_mask.shape[1], :zero_mask.shape[2]],
                                                 np.maximum(one_hot[:, :zero_mask.shape[1], :zero_mask.shape[2]],
                                                  combined_one_hot[:, x1:x2, y1:y2]))

    return combined_one_hot
