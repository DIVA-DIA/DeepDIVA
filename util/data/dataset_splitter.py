"""
This script allows for creation of a validation set from the training set.
"""

# Utils
import argparse
import os
import shutil
import sys
import re
import numpy as np
import random

# Torch related stuff
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split


def split_dataset(dataset_folder, split, symbolic, debug=False):
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
    debug : bool
        Prints additional debug statements

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

    if debug:
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


def _get_file_with_parents(filepath, levels=1):
    common = filepath
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)


def split_dataset_writerIdentification(dataset_folder, split):
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
    print("Data Splitting for Writer Identification\n")
    # Getting the train dir
    binarized_dataset = os.path.join(dataset_folder, "BinarizedDataset")
    colored_dataset = os.path.join(dataset_folder, "ColoredDataset")
    binarized_traindir = os.path.join(binarized_dataset, 'train')
    colored_traindir = os.path.join(colored_dataset, 'train')

    # Rename the original train dir
    shutil.move(binarized_traindir, os.path.join(binarized_dataset, 'original_train'))
    shutil.move(colored_traindir, os.path.join(colored_dataset, 'original_train'))

    binarized_traindir = os.path.join(binarized_dataset, 'original_train')
    colored_traindir = os.path.join(colored_dataset, 'original_train')
    # Sanity check on the training folder
    if not os.path.isdir(binarized_traindir):
        print("Train folder not found in the args.dataset_folder={}".format(binarized_dataset))
        sys.exit(-1)
    if not os.path.isdir(colored_traindir):
        print("Train folder not found in the args.dataset_folder={}".format(colored_dataset))
        sys.exit(-1)

    # Load the dataset file names
    print("Loading dataset filenames\n")
    fileNames = os.listdir(binarized_traindir)
    print("Training set size: :" + str(len(fileNames)))
    validation_size = int(len(fileNames)*split)
    print("Validation set size: " + str(validation_size))
    random.seed(42)
    random.shuffle(fileNames)
    validation_files = random.sample(fileNames, validation_size)
    training_files = [file for file in fileNames if file not in validation_files]

    # Print number of elements for each class
    ''''
    for c in train_binarized_ds.classes:
        print("labels ({}) {}".format(c, np.size(np.where(y_train == train_binarized_ds.class_to_idx[c]))))
    for c in train_binarized_ds.classes:
        print("split_train ({}) {}".format(c, np.size(np.where(y_train == train_binarized_ds.class_to_idx[c]))))
    for c in train_binarized_ds.classes:
        print("split_val ({}) {}".format(c, np.size(np.where(y_val == train_binarized_ds.class_to_idx[c]))))
    '''

    # Create the folder structure to accommodate the two new splits for binarized dataset
    split_train_binarized_dir = os.path.join(binarized_dataset, "train")
    if os.path.exists(split_train_binarized_dir):
        shutil.rmtree(split_train_binarized_dir)
    os.makedirs(split_train_binarized_dir)

    split_train_color_dir = os.path.join(colored_dataset, "train")
    if os.path.exists(split_train_color_dir):
        shutil.rmtree(split_train_color_dir)
    os.makedirs(split_train_color_dir)

    print("Copying files to train folder\n")
    for tf in training_files:
        path_binarized = os.path.join(split_train_binarized_dir, tf)
        path_color = os.path.join(split_train_color_dir, tf)
        if os.path.exists(path_binarized):
            shutil.rmtree(path_binarized)
        os.makedirs(path_binarized)
        if os.path.exists(path_color):
            shutil.rmtree(path_color)
        os.makedirs(path_color)

        binarized_file_path = os.path.join(binarized_traindir, tf)
        subfiles_binarized = os.listdir(binarized_file_path)
        colored_file_path = os.path.join(colored_traindir, tf)
        subfiles_colored = os.listdir(colored_file_path)
        for i in range(len(subfiles_binarized)):
            file = os.path.join(binarized_file_path, subfiles_binarized[i])
            shutil.copy(file, path_binarized)
        for i in range(len(subfiles_colored)):
            file = os.path.join(colored_file_path, subfiles_colored[i])
            shutil.copy(file, path_color)

    split_val_binarized_dir = os.path.join(binarized_dataset, "val")
    if os.path.exists(split_val_binarized_dir):
        shutil.rmtree(split_val_binarized_dir)
    os.makedirs(split_val_binarized_dir)

    split_val_color_dir = os.path.join(colored_dataset, "val")
    if os.path.exists(split_val_color_dir):
        shutil.rmtree(split_val_color_dir)
    os.makedirs(split_val_color_dir)

    print("Copying files to val folder\n")
    for vf in validation_files:

        path_binarized = os.path.join(split_val_binarized_dir, vf)
        path_color = os.path.join(split_val_color_dir, vf)

        if os.path.exists(path_binarized):
            shutil.rmtree(path_binarized)
        os.makedirs(path_binarized)
        if os.path.exists(path_color):
            shutil.rmtree(path_color)
        os.makedirs(path_color)

        binarized_file_path = os.path.join(binarized_traindir, vf)
        subfiles_binarized = os.listdir(binarized_file_path)
        colored_file_path = os.path.join(colored_traindir, vf)
        subfiles_colored = os.listdir(colored_file_path)

        for i in range(len(subfiles_binarized)):
            file = os.path.join(binarized_file_path, subfiles_binarized[i])
            shutil.copy(file, path_binarized)

        for i in range(len(subfiles_colored)):
            file = os.path.join(colored_file_path, subfiles_colored[i])
            shutil.copy(file, path_color)

    print("Splitting is done!")

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

    parser.add_argument('--debug',
                        help='Print additional debug statements.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    split_dataset(dataset_folder=args.dataset_folder, split=args.split, symbolic=args.symbolic)

    split_dataset_writerIdentification(dataset_folder=args.dataset_folder, split=args.split)


