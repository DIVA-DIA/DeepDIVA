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
from util.misc import make_folder_if_not_exists


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


def split_dataset_segmentation(dataset_folder, split, symbolic, test=False):
    """
    Partition a dataset into train/val(/test) splits on the filesystem for segmentation datasets organized
    as dataset/data with the images and dataset/gt for the ground truth. The corresponding images need to have the same
    name.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder (see datasets.image_folder_dataset.load_dataset for details).
    split : float
        Specifies how much of the training set should be converted into the validation set.
    symbolic : bool
        Does not make a copy of the data, but only symbolic links to the original data
    test: bool
        If true, the validation set is split again (1:1) into a val and test set. Default false.

    Returns
    -------
        None
    """
    # Getting the train dir
    orig_dir = os.path.join(dataset_folder, 'train')

    # Rename the original train dir
    shutil.move(orig_dir, os.path.join(dataset_folder, 'original_train'))
    orig_dir = os.path.join(dataset_folder, 'original_train')

    # Sanity check on the training folder
    if not os.path.isdir(orig_dir):
        print("Train folder not found in the args.dataset_folder={}".format(dataset_folder))
        sys.exit(-1)

    # get the dataset splits
    path_data = os.path.join(orig_dir, "data")
    path_gt = os.path.join(orig_dir, "gt")

    file_names_data = sorted(
        [f for f in os.listdir(path_data) if os.path.isfile(os.path.join(path_data, f))])
    file_names_gt = sorted(
        [f for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt, f))])

    # Check data and ensure everything is cool
    assert len(file_names_data) == len(file_names_gt)
    for data, gt in zip(file_names_data, file_names_gt):
        assert data[:-3] == gt[:-3]  # exclude the extension which should be jpg and png
        assert gt[-3:] == "png"

    # Split the data into two sets
    file_names = [(data, gt) for data, gt in zip(file_names_data, file_names_gt)]
    filenames_train, filenames_val, _, _ = train_test_split(file_names, file_names,
                                                            test_size=split,
                                                            random_state=42)

    if test:
        # Split the data into two sets
        filenames_val, filenames_test, _, _ = train_test_split(filenames_val, filenames_val,
                                                               test_size=0.5,
                                                               random_state=42)

    # Make output folders
    dataset_root = os.path.join(dataset_folder)
    train_folder = os.path.join(dataset_root, 'train')
    val_folder = os.path.join(dataset_root, 'val')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(val_folder)

    if test:
        test_folder = os.path.join(dataset_root, 'test')
        make_folder_if_not_exists(test_folder)

    folders = [train_folder, val_folder, test_folder] if test else [train_folder, val_folder]
    file_splits = [filenames_train, filenames_val, filenames_test] if test else [filenames_train, filenames_val]

    # Copying the splits into their folders
    for folder, split_files in zip(folders, file_splits):
        make_folder_if_not_exists(os.path.join(folder, 'data'))
        make_folder_if_not_exists(os.path.join(folder, 'gt'))

        for fdata, fgt in split_files:
            if symbolic:
                os.symlink(os.path.join(path_data, fdata), os.path.join(folder, 'data', fdata))
                os.symlink(os.path.join(path_gt, fgt), os.path.join(folder, 'gt', fgt))

            else:
                shutil.copy(os.path.join(path_data, fdata), os.path.join(folder, 'data', fdata))
                shutil.copy(os.path.join(path_gt, fgt), os.path.join(folder, 'gt', fgt))

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

    parser.add_argument('--test',
                        help='Split val set into half to make a test set.',
                        action='store_true',
                        default=False)

    parser.add_argument('--debug',
                        help='Print additional debug statements.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()


    split_dataset(dataset_folder=args.dataset_folder, split=args.split, symbolic=args.symbolic)

