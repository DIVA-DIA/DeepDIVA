import argparse
import fnmatch
import inspect
import os
import shutil
import sys

import numpy as np
import pandas as pd
import urllib
import zipfile
import re
import csv
import tarfile
import codecs
import gzip
import requests
from tqdm import tqdm
import rarfile
import torch
import torchvision
import wget
from PIL import Image
import scipy
from scipy.io import loadmat as _loadmat
from sklearn.model_selection import train_test_split as _train_test_split

from util.data.dataset_splitter import split_dataset, split_dataset_writerIdentification
from util.misc import get_all_files_in_folders_and_subfolders \
    as _get_all_files_in_folders_and_subfolders, pil_loader, make_folder_if_not_exists


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

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels.detach().numpy())):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
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

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
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

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(os.path.join(args.output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)

def diva_hisdb(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the DIVA HisDB-all dataset for semantic segmentation to the location specified
    on the file system

    See also: https://diuf.unifr.ch/main/hisdoc/diva-hisdb

    Output folder structure: ../HisDB/CB55/train
                             ../HisDB/CB55/val
                             ../HisDB/CB55/test

                             ../HisDB/CB55/test/data -> images
                             ../HisDB/CB55/test/gt   -> pixel-wise annotated ground truth

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # make the root folder
    dataset_root = os.path.join(args.output_folder, 'HisDB')
    make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    link_public = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    link_test_private = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')
    download_path_public = os.path.join(dataset_root, link_public.geturl().rsplit('/', 1)[-1])
    download_path_private = os.path.join(dataset_root, link_test_private.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link_public.geturl()))
    urllib.request.urlretrieve(link_public.geturl(), download_path_public)

    print('Downloading {}...'.format(link_test_private.geturl()))
    urllib.request.urlretrieve(link_test_private.geturl(), download_path_private)
    print('Download complete. Unpacking files...')

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path_public)

    # unpack imgs and gt
    data_gt_zip = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file.namelist() if 'img' in f}
    dataset_folders = [data_file.split('-')[-1][:-4] for data_file in data_gt_zip.keys()]
    for data_file, gt_file in data_gt_zip.items():
        dataset_name = data_file.split('-')[-1][:-4]
        dataset_folder = os.path.join(dataset_root, dataset_name)
        make_folder_if_not_exists(dataset_folder)

        for file in [data_file, gt_file]:
            zip_file.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                # delete zips
                os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for partition in ['train', 'val', 'test', 'test-public']:
            for folder in ['data', 'gt']:
                make_folder_if_not_exists(os.path.join(dataset_folder, partition, folder))

    # move the files to the correct place
    for folder in dataset_folders:
        for k1, v1 in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            for k2, v2 in {'public-test': 'test-public', 'training': 'train', 'validation': 'val'}.items():
                current_path = os.path.join(dataset_root, folder, k1, k2)
                new_path = os.path.join(dataset_root, folder, v2, v1)
                for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                    shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))
            # remove old folders
            shutil.rmtree(os.path.join(dataset_root, folder, k1))

    # fix naming issue
    for old, new in {'CS18': 'CSG18', 'CS863': 'CSG863'}.items():
        os.rename(os.path.join(dataset_root, old), os.path.join(dataset_root, new))

    # unpack private test folders
    zip_file_private = zipfile.ZipFile(download_path_private)

    data_gt_zip_private = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file_private.namelist() if 'img' in f}

    for data_file, gt_file in data_gt_zip_private.items():
        dataset_name = re.search('-(.*)-', data_file).group(1)
        dataset_folder = os.path.join(dataset_root, dataset_name)

        for file in [data_file, gt_file]:
            zip_file_private.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataset_folder, file[:-4]))
            # delete zip
            os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for folder in ['data', 'gt']:
            make_folder_if_not_exists(os.path.join(dataset_folder, 'test', folder))

        for old, new in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            current_path = os.path.join(dataset_folder, "{}-{}-privateTest".format(old, dataset_name), dataset_name)
            new_path = os.path.join(dataset_folder, "test", new)
            for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                # the ground truth files in the private test set have an additional ending, which needs to be remove
                if new == "gt":
                    f_new = re.sub('_gt', r'', f)
                else:
                    f_new = f
                shutil.move(os.path.join(current_path, f), os.path.join(new_path, f_new))

            # remove old folders
            shutil.rmtree(os.path.dirname(current_path))

    print('Finished. Data set up at {}.'.format(dataset_root))

def icdar2017_clamm(args):

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip"
    print("Downloading " + url)
    zip_name = "ICDAR2017_CLaMM_Training.zip"
    local_filename, headers = urllib.request.urlretrieve(url, zip_name)
    zfile = zipfile.ZipFile(local_filename)

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'ICDAR2017-CLAMM')
    dataset_manuscriptDating = os.path.join(dataset_root , 'ManuscriptDating')
    dataset_md_train = os.path.join(dataset_manuscriptDating , 'train')
    dataset_styleClassification = os.path.join(dataset_root , 'StyleClassification')
    dataset_sc_train = os.path.join(dataset_styleClassification, 'train')
    test_sc_folder = os.path.join(dataset_styleClassification, 'test')
    test_md_folder = os.path.join(dataset_manuscriptDating, 'test')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(dataset_manuscriptDating)
    make_folder_if_not_exists(dataset_styleClassification)
    make_folder_if_not_exists(test_sc_folder)

    def _write_data_to_folder(zipfile, filenames, labels, folder, start_index,  isTest):
        print("Writing data\n")
        sorted_labels = [None]*len(labels)
        if isTest == 1:
            for i in range(len(zipfile.infolist())):
                entry = zipfile.infolist()[i]
                if "IRHT_P_009793.tif" in entry.filename:
                    zipfile.infolist().remove(entry)
                    break

        zip_infolist = zipfile.infolist()[1:]

        for i in range(len(zip_infolist)):
            entry = zip_infolist[i]
            entry_index_infilenames = filenames.index(entry.filename[start_index:])
            sorted_labels[i] = labels[entry_index_infilenames]

        for i, (enrty, label) in enumerate(zip(zipfile.infolist()[1:], sorted_labels)):
            with zipfile.open(enrty) as file:
                img = Image.open(file)
                dest = os.path.join(folder, str(label))
                make_folder_if_not_exists(dest)
                img.save(os.path.join(dest, str(i) + '.png'), "PNG", quality=100)

    def getLabels(zfile):
        print("Extracting labels\n")
        filenames, md_labels, sc_labels = [], [], []
        zip_infolist = zfile.infolist()[1:]
        for entry in zip_infolist:
            if '.csv' in entry.filename:
                with zfile.open(entry) as file:
                    cf = file.read()
                    c = csv.StringIO(cf.decode())
                    next(c) # Skip the first line which is the header of csv file
                    for row in c:

                        md_label_strt_ind = row.rfind(';')
                        md_label_end_ind = row.rfind("\r")
                        md_labels.append(row[md_label_strt_ind+1:md_label_end_ind])
                        sc_labels_strt_ind = row[:md_label_strt_ind].rfind(';')
                        sc_labels.append(row[sc_labels_strt_ind+1:md_label_strt_ind])
                        filename_ind = row[:sc_labels_strt_ind].rfind(';')

                        if filename_ind > -1:
                            f_name = row[filename_ind+1:sc_labels_strt_ind]
                        else:
                            f_name = row[:sc_labels_strt_ind]
                        if isTest == 1 and f_name == 'IRHT_P_009783.tif':
                            print('No file named ' + f_name + ". This filename will not be added!")
                        else:
                            filenames.append(f_name)

                zfile.infolist().remove(entry) # remove the csv file from infolist
            if '.db' in entry.filename: # remove the db file from infolist
                zfile.infolist().remove(entry)
        return filenames, sc_labels, md_labels

    isTest = 0
    filenames, sc_labels, md_labels = getLabels(zfile)
    start_index_training = len("ICDAR2017_CLaMM_Training/")
    print("Training data is being prepared for style classification!\n")
    _write_data_to_folder(zfile, filenames, sc_labels, dataset_sc_train, start_index_training, isTest)
    print("Training data is being prepared for manuscript dating!\n")
    _write_data_to_folder(zfile, filenames, md_labels, dataset_md_train, start_index_training, isTest)

    os.remove(os.path.join(zfile.filename))

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_task1_task3.zip"
    print("Downloading " + url)
    zip_name_test = "ICDAR2017_CLaMM_task1_task3.zip"
    local_filename_test, headers_test = urllib.request.urlretrieve(url, zip_name_test)
    zfile_test = zipfile.ZipFile(local_filename_test)

    isTest = 1
    filenames_test, sc_test_labels, md_test_labels = getLabels(zfile_test)
    start_index_test = len("ICDAR2017_CLaMM_task1_task3/")
    print("Test data is being prepared for style classification!\n")
    _write_data_to_folder(zfile_test, filenames_test, sc_test_labels, test_sc_folder, start_index_test, 1)
    print("Test data is being prepared for manuscript dating!\n")
    _write_data_to_folder(zfile_test, filenames_test, md_test_labels, test_md_folder, start_index_test, 1)

    os.remove(os.path.join(zfile_test.filename))
    print("Training-Validation splitting\n")
    split_dataset(dataset_folder=dataset_manuscriptDating, split=0.2, symbolic=False)
    split_dataset(dataset_folder=dataset_styleClassification, split=0.2, symbolic=False)
    print("ICDAR2017 CLaMM data is ready!")


def historical_wi(args):

    train_binarized_url = "ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-training-binarized.zip"
    train_colored_url = "ftp://scruffy.caa.tuwien.ac.at/staff/database/icdar2017/icdar17-historicalwi-training-color.zip"
    test_binarized_url = "https://zenodo.org/record/854353/files/ScriptNet-HistoricalWI-2017-binarized.zip?download=1"
    test_colored_url = "https://zenodo.org/record/854353/files/ScriptNet-HistoricalWI-2017-color.zip?download=1"
    urls = [train_binarized_url, train_colored_url, test_binarized_url, test_colored_url]

    zip_name_train_binarized = "icdar17-historicalwi-training-binarized.zip"
    zip_name_train_color = "icdar17-historicalwi-training-color.zip"
    zip_name_test_binarized = "ScriptNet-HistoricalWI-2017-binarized.zip"
    zip_name_test_color = "ScriptNet-HistoricalWI-2017-color.zip"
    zip_names = [zip_name_train_binarized, zip_name_train_color, zip_name_test_binarized, zip_name_test_color]
    start_indices = [len("icdar2017-training-binary/"), len("icdar2017-training-color/"),
                     len("ScriptNet-HistoricalWI-2017-binarized/"), len("ScriptNet-HistoricalWI-2017-color/")]

    # Make output folders
    """
    dataset_root = os.path.join(args.output_folder)
    train_folder = os.path.join(dataset_root, 'train')
    train_binarized_folder = os.path.join(train_folder, 'Binarized')
    train_colored_folder = os.path.join(train_folder, 'Color')
    test_folder = os.path.join(dataset_root, 'test')
    test_binarized_folder = os.path.join(test_folder, 'Binarized')
    test_colored_folder = os.path.join(test_folder, 'Color')
    folders = [train_binarized_folder, train_colored_folder, test_binarized_folder, test_colored_folder]

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(train_binarized_folder)
    make_folder_if_not_exists(train_colored_folder)
    make_folder_if_not_exists(test_folder)
    make_folder_if_not_exists(test_binarized_folder)
    make_folder_if_not_exists(test_colored_folder)
    """
    dataset_root = os.path.join(os.path.join(args.output_folder, 'historical_wi'))
    binarized_dataset = os.path.join(dataset_root, "BinarizedDataset")
    train_binarized_folder = os.path.join(binarized_dataset, 'train')
    test_binarized_folder = os.path.join(binarized_dataset, 'test')
    colored_dataset = os.path.join(dataset_root, "ColoredDataset")
    train_colored_folder = os.path.join(colored_dataset, 'train')
    test_colored_folder = os.path.join(colored_dataset, 'test')
    folders = [train_binarized_folder, train_colored_folder, test_binarized_folder, test_colored_folder]

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(binarized_dataset)
    make_folder_if_not_exists(colored_dataset)
    make_folder_if_not_exists(train_binarized_folder)
    make_folder_if_not_exists(train_colored_folder)
    make_folder_if_not_exists(test_binarized_folder)
    make_folder_if_not_exists(test_colored_folder)

    def _write_data_to_folder(zipfile, labels, folder, isTrainingset):
        print("Writing data to folder\n")
        for i, (enrty, label) in enumerate(zip(zipfile.infolist()[1:], labels)):
            with zipfile.open(enrty) as file:
                img = Image.open(file)
                dest = os.path.join(folder, str(label))
                make_folder_if_not_exists(dest)
                if isTrainingset == 1:
                    img.save(os.path.join(dest, str(i) + '.png'))
                else:
                    img.save(os.path.join(dest, str(i) + '.jpg'))

    def _get_labels(zipfile, start_index):
        print("Extracting labels\n")
        labels = []
        for zipinfo in zipfile.infolist()[1:]:
            file_name = zipinfo.filename
            ind = file_name.find("-", start_index)
            labels.append(file_name[start_index:ind])
        return labels

    #Prepare Datasets

    for i in range(len(urls)):
        if i < 2:
            isTrainingset = 1
        else:
            isTrainingset = 0

        print("Downloading " + urls[i])
        local_filename, headers = urllib.request.urlretrieve(urls[i], zip_names[i])
        zfile = zipfile.ZipFile(local_filename)
        labels = _get_labels(zfile, start_indices[i])
        _write_data_to_folder(zfile, labels, folders[i], isTrainingset)
        os.remove(os.path.join(zfile.filename))
        if i == 0:
            print("Binary training data is ready!")
        elif i == 1:
            print("Colored training data is ready!")
        elif i == 2:
            print("Binary test data is ready!")
        else:
            print("Colored test data is ready!")

    split_dataset_writerIdentification(dataset_folder=dataset_root, split=0.2)

    print("Historical WI dataset is ready!")


def kmnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the K-MNIST dataset to the location specified
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

    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    def read_image_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert get_int(data[:4]) == 2051
            length = get_int(data[4:8])
            num_rows = get_int(data[8:12])
            num_cols = get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)

    def read_label_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert get_int(data[:4]) == 2049
            length = get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    try:
        torchvision.datasets.KMNIST(root=args.output_folder, download=True)

    except AttributeError:
        url_list = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
                    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
                    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
                    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz']

        raw_folder = os.path.join(args.output_folder, 'raw')
        processed_folder = os.path.join(args.output_folder, 'processed')
        make_folder_if_not_exists(raw_folder)
        make_folder_if_not_exists(processed_folder)

        training_file = 'training.pt'
        test_file = 'test.pt'

        for url in url_list:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(processed_folder, training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(processed_folder, test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'KMNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)
    print("The KMNIST dataset is ready for you at {}".format(dataset_root))


# def kuzushiji_kanji(args):
#     """
#     Fetches and prepares (in a DeepDIVA friendly format) the K-MNIST dataset to the location specified
#     on the file system
#
#     Parameters
#     ----------
#     args : dict
#         List of arguments necessary to run this routine. In particular its necessary to provide
#         output_folder as String containing the path where the dataset will be downloaded
#
#     Returns
#     -------
#         None
#     """
#     url = 'http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar'
#     dataset_root = os.path.join(args.output_folder, 'kkanji')
#
#     path = os.path.join(dataset_root, url.split('/')[-1])
#     r = requests.get(url, stream=True)
#     with open(path, 'wb') as f:
#         total_length = int(r.headers.get('content-length'))
#         print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))
#
#         for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
#             if chunk:
#                 f.write(chunk)
#
#     print('All dataset files downloaded!')
#
#     with tarfile.open(os.path.join(dataset_root, "kkanji.tar")) as f:
#         f.extractall()
#     shutil.rmtree(os.path.join(dataset_root, "kkanji.tar"))
#
#
#     # Make output folders
#     train_folder = os.path.join(dataset_root, 'train')
#     test_folder = os.path.join(dataset_root, 'test')
#
#     make_folder_if_not_exists(dataset_root)
#     make_folder_if_not_exists(train_folder)
#     make_folder_if_not_exists(test_folder)
#
#     split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)
#     print("The kkanji dataset is ready for you at {}".format(dataset_root))


def fashion_mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the Fashion-MNIST dataset to the location specified
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
    torchvision.datasets.FashionMNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'Fashion-MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

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

    make_folder_if_not_exists(dataset_root)
    make_folder_if_not_exists(train_folder)
    make_folder_if_not_exists(val_folder)
    make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(data, labels, folder, classes):
        dest = os.path.join(folder, 'images')
        make_folder_if_not_exists(dest)
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


def glas(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the tubule dataset (from the GlaS challenge) for semantic
    segmentation to the location specified on the file system

    See also: https://github.com/choosehappy/public/tree/master/DL%20tutorial%20Code/3-tubule

    Output folder structure: ../HisDB/GlaS/train
                             ../HisDB/GlaS/val
                             ../HisDB/GlaS/test

                             ../HisDB/GlaS/test/data -> images
                             ../HisDB/GlaS/test/gt   -> pixel-wise annotated ground truth

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """

    def groupby_patient(list_to_group, index=3):
        """
        split images by patient
        :param list_to_group: list of image names
        :param index: position of split by '-' in the image name to obtain patient ID
        :return:  dictionary where keys are patient IDs and values are lists of images that are from that patient
        """
        return {
            '-'.join(filename.split('-')[:index]): [file for file in list_to_group if '-'.join(file.split('-')[:index])
                                                    == '-'.join(filename.split('-')[:index])] for filename in
            list_to_group}

    def convert_gt(img_path):
        img = pil_loader(img_path)

        out_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        out_img[:, :, 2] = 1  # set everything to background in blue channel
        out_img[:, :, 2][img != 0] = 2  # set glands to 2 in blue channel

        out = Image.fromarray(out_img)
        out.save(img_path)

    # make the root folder
    output_folder = args.output_folder
    dataset_root = os.path.join(output_folder, 'GlaS')
    make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    link_tubules = urllib.parse.urlparse(
        'http://andrewjanowczyk.com/wp-static/tubule.tgz')

    download_path_tubules = os.path.join(dataset_root, link_tubules.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link_tubules.geturl()))
    urllib.request.urlretrieve(link_tubules.geturl(), download_path_tubules)

    print('Download complete. Unpacking files...')

    # unpack tubule folder that contains images, annotations and text files with lists of benign and malignant samples
    tar_file = tarfile.open(download_path_tubules)
    tar_file.extractall(path=dataset_root)

    sets_dict = {}
    # 20 benign + 20 malignant images
    train_ids_b = ['09-1339-01',
                   '09-16566-03',
                   '09-21631-03',
                   '09-23232-02',
                   'm9_10741F-12T2N0', '10-13799-05']  # 4*5

    train_ids_m = ['09-322-02',
                   '09-16566-02',
                   '10-13799-06',
                   '10-15247-02',
                   'm6_10719 T3N2a', 'm17_1421 IE-11 T3N2a', 'm18_1421 IE-11 1-86', 'm39_10-1273']  # 5*4

    sets_dict['train'] = train_ids_b + train_ids_m

    # validation has 29 images
    val_ids_b = ['10-12813-05',
                 '10-13799-02',
                 'm2_10449-11E-T3N1b']  # 2*4 + 1 = 9

    val_ids_m = ['09-1339-02',
                 '09-1339-05',
                 '09-1646-01',
                 '09-1646-02',
                 '09-23757-01']  # 5*4 = 20

    sets_dict['val'] = val_ids_b + val_ids_m

    # test has equal mal and ben and 16 img

    test_ids_m = ['09-1646-03', '09-1646-05']  # 2*4 = 8
    test_ids_b = ['10-12813-01', '10-13799-01']  # 2*4 = 8

    sets_dict['test'] = test_ids_b + test_ids_m

    print('Splitting the dataset into train, val and test')
    for s in ['train', 'test', 'val']:
        make_folder_if_not_exists(os.path.join(dataset_root, s, 'gt'))
        make_folder_if_not_exists(os.path.join(dataset_root, s, 'data'))

        print('CREATING {} SET'.format(s))
        for patient in sets_dict[s]:
            for img_file in os.listdir(dataset_root):
                if patient in img_file:
                    if 'anno' in img_file:
                        # convert gt into correct data format
                        convert_gt(os.path.join(dataset_root, img_file))
                        out_file = os.path.join('gt', img_file.replace('_anno', ''))
                    else:
                        out_file = os.path.join('data', img_file)

                    shutil.move(os.path.join(dataset_root, img_file), os.path.join(dataset_root, s, out_file))


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
