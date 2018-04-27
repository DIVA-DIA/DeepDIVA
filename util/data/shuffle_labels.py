"""
This script allows creates a symlink directory with all labels shuffled.

"""

# Utils
import argparse
import os
import shutil
import sys
import random

import numpy as np
# Torch related stuff
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split


def split_dataset(dataset_folder, output_folder, symbolic):
    """ Partition a dataset into train/val splits on the filesystem.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder (see datasets.image_folder_dataset.load_dataset for details).
    output_folder : str
        Path to the output folder (see datasets.image_folder_dataset.load_dataset for details).
    symbolic : bool
        Does not make a copy of the data, but only symbolic links to the original data

    Returns
    -------
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
    fileNames = np.asarray([item[0] for item in train_ds.imgs])
    labels = np.asarray([item[1] for item in train_ds.imgs])

    # Shuffle the labels
    random.shuffle(labels)


    # Create the folder structure to accommodate the two new splits
    split_train_dir = os.path.join(output_folder, "train")
    if os.path.exists(split_train_dir):
        shutil.rmtree(split_train_dir)
    os.makedirs(split_train_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_train_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Copying the splits into their folders
    for X, y in zip(fileNames, labels):
        src = X
        file_name = os.path.basename(src)
        dest = os.path.join(split_train_dir, train_ds.classes[y], file_name)
        if symbolic:
            os.symlink(src, dest)
        else:
            shutil.copy(X, dest)

    # Symlink val/test to train
    os.symlink(split_train_dir, os.path.join(output_folder, 'val'))
    os.symlink(split_train_dir, os.path.join(output_folder, 'test'))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script creates a dataset with randomly '
                                                 'shuffled labels.'
                                                 'WARNING: DO NOT USE IF YOU DON\'T KNOW '
                                                 'WHAT THIS MEANS!')

    parser.add_argument('--dataset-folder',
                        help='path to root of the dataset.',
                        required=True,
                        type=str,
                        default=None)

    parser.add_argument('--output-folder',
                        help='path to where symlink directory should be created.',
                        required=True,
                        type=str,
                        default=None)

    parser.add_argument('--symbolic',
                        help='Make symbolic links instead of copies.',
                        action='store_false',
                        default=True)

    args = parser.parse_args()

    split_dataset(dataset_folder=args.dataset_folder, output_folder=args.output_folder, symbolic=args.symbolic)
