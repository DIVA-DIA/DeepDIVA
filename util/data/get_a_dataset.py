# Utils
import argparse
import inspect
import os
import shutil
import sys
from PIL import Image

# Torch
import torch
import torchvision

# DeepDIVA
from util.data.dataset_splitter import split_dataset


def mnist(args):
    """
    Fetches and prepare in a DeepDIVA friendly format the MNIST dataset on the location specified
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
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
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

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('dataset',
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
