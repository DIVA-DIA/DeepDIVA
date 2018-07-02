"""
This script perform some analysis on the dataset provided.
In particular computes std and mean (to be used to center your dataset).

Structure of the dataset expected:

Split folders
-------------
'args.dataset-folder' has to point to the parent of the train folder.
Example:

        ~/../../data/svhn

where the dataset_folder contains the train sub-folder as follow:

    args.dataset_folder/train

Classes folders
---------------
The train split should have different classes in a separate folder with the class
name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).
Example:

    train/dog/whatever.png
    train/dog/you.png
    train/dog/like.png

    train/cat/123.png
    train/cat/nsdf3.png
    train/cat/asd932_.png

@author: Michele Alberti
"""

# Utils
import argparse
import logging
import os
import sys
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd

# Torch related stuff
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def compute_mean_std(dataset_folder, inmem, workers):
    """
    Computes mean and std of a dataset. Saves the results as CSV file in the dataset folder.

    Parameters
    ----------
    dataset_folder : String (path)
        Path to the dataset folder (see above for details)
    inmem : Boolean
        Specifies whether is should be computed i nan online of offline fashion.
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        logging.warning("Train folder not found in the args.dataset_folder={}".format(dataset_folder))
        return

    # Load the dataset file names
    train_ds = datasets.ImageFolder(traindir,
                                    transform=transforms.Compose([transforms.ToTensor()]))

    # Extract the actual file names and labels as entries
    file_names = np.asarray([item[0] for item in train_ds.imgs])

    # Compute mean and std
    if inmem:
        mean, std = cms_inmem(file_names)
    else:
        mean, std = cms_online(file_names, workers)

    # Compute class frequencies weights
    class_frequencies_weights = _get_class_frequencies_weights(train_ds, workers)

    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std, class_frequencies_weights])
    df.index = ['mean[RGB]', 'std[RGB]', 'class_frequencies_weights[num_classes]']
    df.to_csv(os.path.join(dataset_folder, 'analytics.csv'), header=False)


# Loads an image with OpenCV and returns the channel wise means of the image.
def _return_mean(image_path):
    # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
    img = cv2.imread(image_path)
    mean = np.array([np.mean(img[:, :, 2]), np.mean(img[:, :, 1]), np.mean(img[:, :, 0])]) / 255.0
    return mean


# Loads an image with OpenCV and returns the
def _return_std(image_path, mean):
    # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
    img = cv2.imread(image_path) / 255.0
    m2 = np.square(np.array([img[:, :, 2] - mean[0], img[:, :, 1] - mean[1], img[:, :, 0] - mean[2]]))
    return np.sum(np.sum(m2, axis=1), 1), m2.size / 3.0


def cms_online(file_names, workers):
    """
    Computes mean and image_classification deviation in an online fashion. This is useful when the dataset is too big to
    be allocated in memory. 

    Parameters
    ----------
    file_names : List of String
        List of file names of the dataset
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    mean : double
    std : double
    """

    # Set up a pool of workers
    pool = Pool(workers)

    logging.info('Begin computing the mean')

    # Online mean
    results = pool.map(_return_mean, file_names)
    mean_sum = np.sum(np.array(results), axis=0)

    # Divide by number of samples in train set
    mean = mean_sum / file_names.size

    logging.info('Finished computing the mean')
    logging.info('Begin computing the std')

    # Online image_classification deviation
    results = pool.starmap(_return_std, [[item, mean] for item in file_names])
    std_sum = np.sum(np.array([item[0] for item in results]), axis=0)
    total_pixel_count = np.sum(np.array([item[1] for item in results]))
    std = np.sqrt(std_sum / total_pixel_count)
    logging.info('Finished computing the std')

    # Shut down the pool
    pool.close()

    return mean, std


def cms_inmem(file_names):
    """
    Computes mean and image_classification deviation in an offline fashion. This is possible only when the dataset can
    be allocated in memory.

    Parameters
    ----------
    file_names: List of String
        List of file names of the dataset
    Returns
    -------
    mean : double
    std : double
    """
    img = np.zeros([file_names.size] + list(cv2.imread(file_names[0]).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = cv2.imread(sample)

    # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
    mean = np.array([np.mean(img[:, :, :, 2]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 0])]) / 255.0
    std = np.array([np.std(img[:, :, :, 2]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 0])]) / 255.0

    return mean, std


def _get_class_frequencies_weights(dataset, workers):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    train_loader: torch.utils.data.dataloader.DataLoader
        Dataloader for the training se
    workers: int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array normalized (sum up to 1)
    """
    logging.info('Begin computing class frequencies weights')
    all_labels = None
    try:
        all_labels = [item[1] for item in dataset.imgs]
    except:
        all_labels = [item for item in dataset.labels]
    finally:
        if all_labels == None:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=workers)
            all_labels = []
            for target, label in data_loader:
                all_labels.append(label)
            all_labels = np.concatenate(all_labels).reshape(len(dataset))

    total_num_samples = len(all_labels)
    num_samples_per_class = np.unique(all_labels, return_counts=True)[1]
    class_frequencies = (num_samples_per_class / total_num_samples)
    logging.info('Finished computing class frequencies weights')
    logging.info('Class frequencies (rounded): {class_frequencies}'
                 .format(class_frequencies=np.around(class_frequencies * 100, decimals=2)))
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    return (1 / num_samples_per_class) / ((1 / num_samples_per_class).sum())


if __name__ == "__main__":
    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script perform some analysis on the dataset provided')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        type=str)

    parser.add_argument('--online',
                        action='store_true',
                        help='Compute it in an online fashion (because it probably will not fin in memory')

    args = parser.parse_args()

    compute_mean_std(dataset_folder=args.dataset_folder,
                     inmem=args.online)
