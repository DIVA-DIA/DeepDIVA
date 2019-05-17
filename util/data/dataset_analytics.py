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
"""

# Utils
import argparse
import logging
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image

# Torch related stuff
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from util.misc import load_numpy_image

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
    train_ds = datasets.ImageFolder(traindir, transform=transforms.Compose([transforms.ToTensor()]))

    # Extract the actual file names and labels as entries
    file_names = np.asarray([item[0] for item in train_ds.imgs])

    # Compute mean and std
    if inmem:
        mean, std = cms_inmem(file_names)
    else:
        mean, std = cms_online(file_names, workers)

    # Check if the dataset is a multi-label dataset
    if not os.path.exists(os.path.join(traindir, 'labels.csv')):
        # Use normal class frequency computation
        class_frequencies_weights = _get_class_frequencies_weights(train_ds, workers)
    else:
        # Use multi-label class frequency computation
        class_frequencies_weights = _get_class_frequencies_weights_multilabel(os.path.join(traindir, 'labels.csv'))

    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std, class_frequencies_weights])
    df.index = ['mean[RGB]', 'std[RGB]', 'class_frequencies_weights[num_classes]']
    df.to_csv(os.path.join(dataset_folder, 'analytics.csv'), header=False)


def compute_mean_std_segmentation(dataset_folder, inmem, workers):
    """
    Computes mean and std of a dataset for semantic segmentation. Saves the results as CSV file in the dataset folder.

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

    # Load the dataset file names
    train_ds = datasets.ImageFolder(traindir, transform=transforms.Compose([transforms.ToTensor()]))

    # Extract the actual file names and labels as entries
    file_names_all = np.asarray([item[0] for item in train_ds.imgs])
    file_names_gt = np.asarray([f for f in file_names_all if '/gt/' in f])
    file_names_data = np.asarray([f for f in file_names_all if '/data/' in f])

    # Compute mean and std
    if inmem:
        mean, std = cms_inmem(file_names_data)
    else:
        mean, std = cms_online(file_names_data, workers)

    # Compute class frequencies weights
    class_frequencies_weights, class_ints = _get_class_frequencies_weights_segmentation(file_names_gt)
    # print(class_frequencies_weights)
    # Save results as CSV file in the dataset folder
    df = pd.DataFrame([mean, std, class_frequencies_weights, class_ints])
    df.index = ['mean[RGB]', 'std[RGB]', 'class_frequencies_weights[num_classes]', 'class_encodings']
    df.to_csv(os.path.join(dataset_folder, 'analytics.csv'), header=False)


# Loads an image with OpenCV and returns the channel wise means of the image.
def _return_mean(image_path):
    img = load_numpy_image(image_path)
    mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])]) / 255.0
    return mean


# Loads an image with OpenCV and returns the channel wise std of the image.
def _return_std(image_path, mean):
    img = load_numpy_image(image_path) / 255.0
    m2 = np.square(np.array([img[:, :, 0] - mean[0], img[:, :, 1] - mean[1], img[:, :, 2] - mean[2]]))
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
    img = np.zeros([file_names.size] + list(load_numpy_image(file_names[0]).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = load_numpy_image(sample)

    mean = np.array([np.mean(img[:, :, :, 0]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 2])]) / 255.0
    std = np.array([np.std(img[:, :, :, 0]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 2])]) / 255.0

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


def _get_class_frequencies_weights_segmentation(gt_images):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    gt_images: list of strings
        Path to all ground truth images, which contain the pixel-wise label
    workers: int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes) and ints the classes are represented as
        The weights vector as a 1D array normalized (sum up to 1)
    """
    logging.info('Begin computing class frequencies weights')

    total_num_pixels = 0
    label_counter = {}

    for path in gt_images:
        img = np.array(Image.open(path))[:, :, 2].flatten()
        total_num_pixels += len(img)
        for i, j in zip(*np.unique(img, return_counts=True)):
            label_counter[i] = label_counter.get(i, 0) + j

    classes = np.array(sorted(label_counter.keys()))
    num_samples_per_class = np.array([label_counter[k] for k in classes])
    class_frequencies = (num_samples_per_class / total_num_pixels)
    logging.info('Finished computing class frequencies weights')
    logging.info('Class frequencies (rounded): {class_frequencies}'
                 .format(class_frequencies=np.around(class_frequencies * 100, decimals=2)))
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    return (1 / num_samples_per_class) / ((1 / num_samples_per_class).sum()), classes


def _get_class_frequencies_weights_coco(dataset, name_onehotindex, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    dataset: pycocotools.coco.COCO
        COCO dataset loaded with the pycocotools and the torchvision dataset loader

    classes: dict
        dictionary containing the class names and the corresponding index for argmax

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array normalized (sum up to 1)
    """
    logging.info('Begin computing class frequencies weights')

    count_labels = {v: 0 for v in name_onehotindex.values()}

    for (_, gt_mask) in dataset:
        for k, v in zip(*np.unique(np.array(gt_mask).flatten(), return_counts=True)):
            count_labels[k] += v

    total_num_samples = sum(count_labels.values())
    num_samples_per_class = np.array([count_labels[k] for k in sorted(count_labels.keys())])
    class_frequencies = (num_samples_per_class / total_num_samples)
    logging.info('Finished computing class frequencies weights')
    logging.info('Class frequencies (rounded): {class_frequencies}'
                 .format(class_frequencies=np.around(class_frequencies * 100, decimals=2)))
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    return (1 / num_samples_per_class) / ((1 / num_samples_per_class).sum())


def _get_class_frequencies_weights_multilabel(dataset_labels):
    """
    Computes the weights for each class (as required by torch.nn.BCEWithLogitsLoss).
    The weight for each class is #neg_samples/#pos_samples.

    Parameters
    ----------
    dataset_folder: torch.utils.data.dataloader.DataLoader
        Path to a labels.csv file with labels for each training sample

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array
    """
    logging.info('Begin computing class weights')

    labels_df = pd.read_csv(dataset_labels)
    classes = labels_df.columns
    labels = labels_df.values

    # Replace all -1 with 0
    labels[labels == -1] = 0

    # Remove the filenames
    labels = labels[:, 1:]

    weights = []
    for i in range(len(labels[0])):
        pos = len(np.where(labels[:, i] == 1)[0])
        neg = len(labels) - pos
        weight = neg / pos
        weights.append(weight)

    weights = np.array(weights)

    logging.info('Finished computing class weights')
    logging.info('Class weights (rounded): {}'.format(np.around(weights, decimals=2)))
    return weights


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        level=logging.INFO
    )

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
