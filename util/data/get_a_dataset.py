
import argparse
import inspect
import os
import shutil
import sys

import numpy as np
import pandas as pd
import rarfile
import torch
import torchvision
import wget
from PIL import Image
from scipy.io import loadmat as _loadmat
from sklearn.model_selection import train_test_split as _train_test_split

from util.data.dataset_splitter import split_dataset
from util.misc import get_all_files_in_folders_and_subfolders \
    as _get_all_files_in_folders_and_subfolders


def mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'MNIST',
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'MNIST',
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'MNIST', 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'MNIST', 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def svhn(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the SVHN dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.SVHN(root=args.output_folder, split='train', download=True)
    torchvision.datasets.SVHN(root=args.output_folder, split='test', download=True)

    # Load the data into memory
    train = _loadmat(os.path.join(args.output_folder,
                                  'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = _loadmat(os.path.join(args.output_folder,
                                 'test_32x32.mat'))
    test_data, test_labels = test['X'], test['y'].astype(np.int64).squeeze()
    np.place(test_labels, test_labels == 10, 0)
    test_data = np.transpose(test_data, (3, 0, 1, 2))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'SVHN')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'train_32x32.mat'))
    os.remove(os.path.join(args.output_folder, 'test_32x32.mat'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def cifar10(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the CIFAR dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    cifar_train = torchvision.datasets.CIFAR10(root=args.output_folder, train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root=args.output_folder, train=False, download=True)

    # Load the data into memory
    train_data, train_labels = cifar_train.data, cifar_train.targets

    test_data, test_labels = cifar_test.data, cifar_test.targets

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'CIFAR10')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(os.path.join(args.output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def miml(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the Multi-Instance Multi-Label Image Dataset
    on the file system. Dataset available at: http://lamda.nju.edu.cn/data_MIMLimage.ashx

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Download the files
    url = 'http://lamda.nju.edu.cn/files/miml-image-data.rar'
    if not os.path.exists(os.path.join(args.output_folder, 'miml-image-data.rar')):
        print('Downloading file!')
        filename = wget.download(url, out=args.output_folder)
    else:
        print('File already downloaded!')
        filename = os.path.join(args.output_folder, 'miml-image-data.rar')

    # Extract the files
    path_to_rar = filename
    path_to_output = os.path.join(args.output_folder, 'tmp_miml')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    path_to_rar = os.path.join(path_to_output, 'original.rar')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    path_to_rar = os.path.join(path_to_output, 'processed.rar')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    print('Extracted files...')

    # Load the mat file
    mat = _loadmat(os.path.join(path_to_output, 'miml data.mat'))
    targets = mat['targets'].T
    classes = [item[0][0] for item in mat['class_name']]
    # Add filename at 0-index to correctly format the CSV headers
    classes.insert(0, 'filename')

    # Get list of all image files in the folder
    images = [item for item in _get_all_files_in_folders_and_subfolders(path_to_output)
              if item.endswith('jpg')]
    images = sorted(images, key=lambda e: int(os.path.basename(e).split('.')[0]))

    # Make splits
    train_data, test_data, train_labels, test_labels = _train_test_split(images, targets, test_size=0.2,
                                                                         random_state=42)
    train_data, val_data, train_labels, val_labels = _train_test_split(train_data, train_labels, test_size=0.2,
                                                                       random_state=42)

    # print('Size of splits\ntrain:{}\nval:{}\ntest:{}'.format(len(train_data),
    #                                                     len(val_data),
    #                                                     len(test_data)))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MIML')
    train_folder = os.path.join(dataset_root, 'train')
    val_folder = os.path.join(dataset_root, 'val')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(val_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(data, labels, folder, classes):
        dest = os.path.join(folder, 'images')
        _make_folder_if_not_exists(dest)
        for image, label in zip(data, labels):
            shutil.copy(image, dest)

        rows = np.column_stack(([os.path.join('images', os.path.basename(item)) for item in data], labels))
        rows = sorted(rows, key=lambda e: int(e[0].split('/')[1].split('.')[0]))
        output_csv = pd.DataFrame(rows)
        output_csv.to_csv(os.path.join(folder, 'labels.csv'), header=classes, index=False)
        return

    # Write the images to the correct folders
    print('Writing the data to the filesystem')
    _write_data_to_folder(train_data, train_labels, train_folder, classes)
    _write_data_to_folder(val_data, val_labels, val_folder, classes)
    _write_data_to_folder(test_data, test_labels, test_folder, classes)

    os.remove(filename)
    shutil.rmtree(path_to_output)
    print('All done!')
    return


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
