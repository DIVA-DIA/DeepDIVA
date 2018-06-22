"""
This script allows for creation of a validation set from the training set.
"""

# Utils
import argparse
import os
import shutil
import sys
import numpy as np

# Torch related stuff
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split


def split_dataset(dataset_folder, split, symbolic):
    """
    Partition a dataset into train/val splits on the filesystem.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder (see datasets.image_folder_dataset.load_dataset for details).
    split : float
        Specifies how much of the training set should be converted into the validation set.
    symbolic : bool
        Does not make a copy of the data, but only symbolic links to the original data

    Returns
    -------
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Rename the original train dir
    shutil.move(traindir, os.path.join(dataset_folder, 'original_train'))
    traindir = os.path.join(dataset_folder, 'original_train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        print("Train folder not found in the args.dataset_folder={}".format(dataset_folder))
        sys.exit(-1)

    # Load the dataset file names

    train_ds = datasets.ImageFolder(traindir)

    # Extract the actual file names and labels as entries
    fileNames = np.asarray([item[0] for item in train_ds.imgs])
    labels = np.asarray([item[1] for item in train_ds.imgs])

    # Split the data into two sets
    X_train, X_val, y_train, y_val = train_test_split(fileNames, labels,
                                                      test_size=split,
                                                      random_state=42,
                                                      stratify=labels)

    # Print number of elements for each class
    for c in train_ds.classes:
        print("labels ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_train ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_val ({}) {}".format(c, np.size(np.where(y_val == train_ds.class_to_idx[c]))))

    # Create the folder structure to accommodate the two new splits
    split_train_dir = os.path.join(dataset_folder, "train")
    if os.path.exists(split_train_dir):
        shutil.rmtree(split_train_dir)
    os.makedirs(split_train_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_train_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    split_val_dir = os.path.join(dataset_folder, "val")
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
        src = X
        file_name = os.path.basename(src)
        dest = os.path.join(split_train_dir, train_ds.classes[y], file_name)
        if symbolic:
            os.symlink(src, dest)
        else:
            shutil.copy(X, dest)

    for X, y in zip(X_val, y_val):
        src = X
        file_name = os.path.basename(src)
        dest = os.path.join(split_val_dir, train_ds.classes[y], file_name)
        if symbolic:
            os.symlink(src, dest)
        else:
            shutil.copy(X, dest)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script creates train/val splits '
                                                 'from a specified dataset folder.')

    parser.add_argument('--dataset-folder',
                        help='path to root of the dataset.',
                        required=True,
                        type=str,
                        default=None)

    parser.add_argument('--split',
                        help='Ratio of the split for validation set.'
                             'Example: if 0.2 the training set will be 80% and val 20%.',
                        type=float,
                        default=0.2)

    parser.add_argument('--symbolic',
                        help='Make symbolic links instead of copies.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    split_dataset(dataset_folder=args.dataset_folder, split=args.split, symbolic=args.symbolic)
