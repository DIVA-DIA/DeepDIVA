"""
This script allows for creation of a validation set from the training set.

Structure of the dataset expected:

Split folders
-------------
'args.dataset-folder' has to point to the parent of the train folder.
Example:

        ~/../../data/svhn

where the dataset_folder contains the train sub-folder as follow:

        args.dataset_folder/train

after running this script there will be the two splits sets follow:

        args.dataset_folder/split_train
        args.dataset_folder/split_val

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
import shutil
import sys

# Torch related stuff
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

# DeepDIVA
from init.initializer import *


def split_dataset(dataset_folder, split):
    """

    Parameters
    ----------
    :param dataset_folder: String (path)
        Path to the dataset folder (see above for details)

    :param split: double ]0,1[
        Specifies how much of the training set should be converted into the validation set

    :return:
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # Load the dataset file names

    train_ds = datasets.ImageFolder(traindir)

    # Extract the actual file names and labels as entries
    fileNames = np.asarray([item[0] for item in train_ds.imgs])
    labels = np.asarray([item[1] for item in train_ds.imgs])

    # Split the data into two sets
    X_train, X_val, y_train, y_val = train_test_split(fileNames, labels, test_size=split, random_state=42)

    # Print number of elements for teach class
    for c in train_ds.classes:
        print("labels ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_train ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_val ({}) {}".format(c, np.size(np.where(y_val == train_ds.class_to_idx[c]))))

    # Create the folder structure to accommodate the two new splits
    split_train_dir = os.path.join(dataset_folder, "split_train")
    if os.path.exists(split_train_dir):
        shutil.rmtree(split_train_dir)
    os.makedirs(split_train_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_train_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    split_val_dir = os.path.join(dataset_folder, "split_val")
    if os.path.exists(split_val_dir):
        shutil.rmtree(split_val_dir)
    os.makedirs(split_val_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_val_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Copying the splits into their folders
    for X, y in zip(X_train, y_train):
        path = os.path.join(split_train_dir, train_ds.classes[y])
        shutil.copy(X, path)

    for X, y in zip(X_val, y_val):
        path = os.path.join(split_val_dir, train_ds.classes[y])
        shutil.copy(X, path)


if __name__ == "__main__":
    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script allows for creation of a validation set from the training set')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        default=None,
                        type=str)

    parser.add_argument('--split',
                        help='Ratio of the split for validation set.'
                             'Example: if 0.2 the training set will be 80% and val 20%.',
                        type=float,
                        default=0.2)

    args = parser.parse_args()

    split_dataset(dataset_folder=args.dataset_folder, split=args.split)
