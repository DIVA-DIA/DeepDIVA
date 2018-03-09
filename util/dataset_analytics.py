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
import os
import sys

import cv2
import numpy as np
import pandas as pd
# Torch related stuff
import torchvision.datasets as datasets


def compute_mean_std(dataset_folder, inmem=False):
    """
    Computes mean and std of a dataset. Saves the results as CSV file in the dataset folder.

    Parameters
    ----------
    :param dataset_folder: String (path)
        Path to the dataset folder (see above for details)

    :param inmem: Boolean
        Specifies whether is should be computed i nan online of offline fashion.

    :return:
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        print("Train folder not found in the args.dataset_folder={}".format(dataset_folder))
        sys.exit(-1)

    # Load the dataset file names
    train_ds = datasets.ImageFolder(traindir)

    # Extract the actual file names and labels as entries
    file_names = np.asarray([item[0] for item in train_ds.imgs])

    # Compute mean and std
    if not inmem:
        mean, std = cms_online(file_names)
    else:
        mean, std = cms_offline(file_names)

    # Display the results on console
    print("Mean: [{}, {}, {}]".format(mean[0], mean[1], mean[2]))
    print("Std: [{}, {}, {}]".format(std[0], std[1], std[2]))

    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std])
    df.index = ['mean[RGB]', 'std[RGB]']
    df.to_csv(os.path.join(dataset_folder, 'analytics.csv'), header=False)


def cms_online(file_names):
    """
    Computes mean and standard deviation in an online fashion. This is useful when the dataset is too big to
    be allocated in memory. 

    Parameters
    ----------
    :param file_names: List of String
        List of file names of the dataset
    :return:
        Mean (double) and Std (double)
    """
    # Online mean
    mean = [0, 0, 0]
    for sample in file_names:
        # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
        img = cv2.imread(sample)
        mean += np.array([np.mean(img[:, :, 2]), np.mean(img[:, :, 1]), np.mean(img[:, :, 0])]) / 255.0

    # Divide by number of samples in train set
    mean /= file_names.size

    # Online standard deviation
    std = [0, 0, 0]
    for sample in file_names:
        # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
        img = cv2.imread(sample) / 255.0
        m2 = np.square(np.array([img[:, :, 2] - mean[0], img[:, :, 1] - mean[1], img[:, :, 0] - mean[2]]))
        std += np.sum(np.sum(m2, axis=1), axis=1)

    std = np.sqrt(std / (file_names.size * (m2.size / 3.0)))
    return mean, std


def cms_offline(file_names):
    """
    Computes mean and standard deviation in an offline fashion. This is possible only when the dataset can
    be allocated in memory.

    Parameters
    ----------
    :param file_names: List of String
        List of file names of the dataset
    :return:
        Mean (double) and Std (double)
    """
    img = np.zeros([file_names.size] + list(cv2.imread(file_names[0]).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = cv2.imread(sample)

    # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
    mean = np.array([np.mean(img[:, :, :, 2]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 0])]) / 255.0
    std = np.array([np.std(img[:, :, :, 2]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 0])]) / 255.0

    return mean, std


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
