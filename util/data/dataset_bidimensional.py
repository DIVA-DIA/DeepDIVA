"""
This script allows for creation of a bidimensional(2D) dataset.
"""

# Utils
import argparse
import inspect
import logging
import os
import random
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def diagonal(size):
    """
    Generates a dataset where points on a diagonal line are one class,
    and points surrounding it are a different class.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    # Generate data
    samples = np.array([(x, y, 0 if x > y else 1)
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def circle(size):
    """
    Samples are generated in a grid fashion (np.linspace) and then draw a circle on x*x + y*y > 0.5
    2 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    # Compute center point lying on the grid of np.linspace (0.5050505050501)
    mid_pt = np.linspace(0, 1, np.sqrt(size))
    mid_pt = mid_pt[int(len(mid_pt) / 2)]

    samples = np.array([(x, y, 0 if (x - mid_pt) ** 2 + (y - mid_pt) ** 2 < 0.15 else 1)
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def donut(size):
    """
    Samples are generated in a grid fashion (np.linspace) and then draw a donut.
    2 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    # Generate data
    mid_pt = np.linspace(0, 1, np.sqrt(size))
    mid_pt = mid_pt[int(len(mid_pt) / 2)]

    samples = np.array([(x, y, 0 if (0.15 > (x - mid_pt) ** 2 + (y - mid_pt) ** 2 > 0.10) else 1)
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def stripes(size):
    """
    Samples are generated in a stripe fashion, like a TV color screen (vertical stripes). Each bin is a different class.
    5 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """
    # The *0.99 serves to make the points on 1.0 to "fall on the left bin" otherwise you get 1 more class
    samples = np.array([(x, y, int((x * 0.99 * 100) / 20))
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def spiral(size):
    """
    Samples are generated in a two spiral fashion, starting from the center.
    2 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    turn_factor = 12

    samples = np.zeros((2 * size, 3))

    for n in range(size):
        r = 0.05 + 0.4 * n / size
        angle = r * turn_factor * np.math.pi
        samples[n] = [0.5 + r * np.math.cos(angle), 0.5 + r * np.math.sin(angle), 0]
        angle = r * turn_factor * np.math.pi + np.math.pi
        samples[n + size] = [0.5 + r * np.math.cos(angle), 0.5 + r * np.math.sin(angle), 1]

    return _split_data(samples)


def spiral_multi(size):
    """
    Samples are generated in a two spiral fashion, starting from the center.
    4 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    turn_factor = -4
    noise = 0.07

    samples = np.zeros((4 * size, 3))

    for n in range(size):
        r = 0.05 + 0.4 * n / size
        # Class 1
        angle = r * turn_factor * np.math.pi
        samples[n + 0 * size] = [0.5 + r * np.math.cos(angle) + random.random() * noise,
                                 0.5 + r * np.math.sin(angle) + random.random() * noise,
                                 0]
        # Class 2
        angle = r * turn_factor * np.math.pi + np.math.pi * 2 / 4.0
        samples[n + 1 * size] = [0.5 + r * np.math.cos(angle) + random.random() * noise,
                                 0.5 + r * np.math.sin(angle) + random.random() * noise,
                                 1]
        # Class 3
        angle = r * turn_factor * np.math.pi + np.math.pi * 4 / 4.0
        samples[n + 2 * size] = [0.5 + r * np.math.cos(angle) + random.random() * noise,
                                 0.5 + r * np.math.sin(angle) + random.random() * noise,
                                 2]
        # Class 4
        angle = r * turn_factor * np.math.pi + np.math.pi * 6 / 4.0
        samples[n + 3 * size] = [0.5 + r * np.math.cos(angle) + random.random() * noise,
                                 0.5 + r * np.math.sin(angle) + random.random() * noise,
                                 3]
    return _split_data(samples)


def xor(size):
    """
    XOR problem
    2 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    samples = np.array([(x, y, ((x < 0.5) and (y < 0.5)) or ((x > 0.5) and (y > 0.5)))
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def flag(size):
    """
    XOR problem but with multi class, each corner a different class
    4 classes.

    Parameters
    ----------
    size : int
        The total number of points in the dataset.
    
    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """

    samples = np.array([(x, y, _multi_quadrant(x, y))
                        for x in np.linspace(0, 1, np.sqrt(size))
                        for y in np.linspace(0, 1, np.sqrt(size))])

    return _split_data(samples)


def _multi_quadrant(x, y):
    if (x < 0.5) and (y < 0.5):
        return 0
    if (x < 0.5) and (y > 0.5):
        return 1
    if (x > 0.5) and (y < 0.5):
        return 2
    if (x > 0.5) and (y > 0.5):
        return 3


########################################################################################################################
def _split_data(samples):
    """
    Split the given samples array into train validation and test sets with ratio 6, 2, 2

    Parameters
    ----------
    samples : np.array(n,m+1)
        The samples to be split: n is the number of samples, m is the number of dimensions and the +1 is the label

    Returns
    -------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    """
    # Split it
    train, tmp, label_train, label_tmp = train_test_split(samples[:, 0:2], samples[:, 2], test_size=0.4,
                                                          random_state=42)
    val, test, label_val, label_test = train_test_split(tmp, label_tmp, test_size=0.5, random_state=42)

    # Return the different splits by selecting x,y from the data and the relative label
    return np.array([[a[0], a[1], b] for a, b in zip(train, label_train)]), \
           np.array([[a[0], a[1], b] for a, b in zip(val, label_val)]), \
           np.array([[a[0], a[1], b] for a, b in zip(test, label_test)])


def _visualize_distribution(train, val, test, save_path, marker_size=1):
    """
    This routine creates a PDF with three images for train, val and test respectively where
    each image is a visual representation of the split distribution with class colors.

    Parameters
    ----------
    train, val, test : ndarray[float] of size (n,3)
        The three splits. Each row is (x,y,label)
    save_path : String
        Path where to save the PDF
    marker_size : float
        Size of the marker representing each datapoint. For big dataset make this small

    Returns
    -------
        None
    """
    fig, axs = plt.subplots(ncols=3, sharex='all', sharey='all')
    plt.setp(axs.flat, aspect=1.0, adjustable='box-forced')
    axs[0].scatter(train[:, 0], train[:, 1], c=train[:, 2], s=marker_size, cmap=plt.get_cmap('Set1'))
    axs[0].set_title('train')
    axs[1].scatter(val[:, 0], val[:, 1], c=val[:, 2], s=marker_size, cmap=plt.get_cmap('Set1'))
    axs[1].set_title('val')
    axs[2].scatter(test[:, 0], test[:, 1], c=test[:, 2], s=marker_size, cmap=plt.get_cmap('Set1'))
    axs[2].set_title('test')
    fig.canvas.draw()
    fig.savefig(save_path)
    fig.clf()
    plt.close()


if __name__ == "__main__":

    # Distribution options:
    distribution_options = [name[0] for name in inspect.getmembers(sys.modules[__name__], inspect.isfunction)]

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        level=logging.INFO)

    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script allows for creation of a validation set from the training set')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        type=str)

    parser.add_argument('--distribution',
                        help='Kind of distribution of the points',
                        choices=distribution_options,
                        required=True,
                        type=str)

    parser.add_argument('--size',
                        help='Total amount of samples.',
                        type=int,
                        default=100)

    args = parser.parse_args()

    ###############################################################################
    # Getting the data
    logging.info('Getting the data distribution {}'.format(args.distribution))
    train, val, test = getattr(sys.modules[__name__], args.distribution)(args.size)

    ###############################################################################
    # Preparing the folders structure

    # Sanity check on the dataset folder
    logging.info('Sanity check on the dataset folder')
    if not os.path.isdir(args.dataset_folder):
        print("Dataset folder not found in the args.dataset_folder={}".format(args.dataset_folder))
        sys.exit(-1)

    # Creating the folder for the dataset
    logging.info('Creating the folder for the dataset')
    dataset_dir = os.path.join(args.dataset_folder, 'bd_' + args.distribution)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    # Creating the folders for the splits
    logging.info('Creating the folders for the splits')
    train_dir = os.path.join(dataset_dir, 'train')
    os.makedirs(train_dir)

    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(val_dir)

    test_dir = os.path.join(dataset_dir, 'test')
    os.makedirs(test_dir)

    ###############################################################################
    # Save splits on csv format with n rows where each row is (x,y,label)
    logging.info('Save splits on csv format')
    pd.DataFrame(train).to_csv(os.path.join(train_dir, 'data.csv'), index=False, header=False)
    pd.DataFrame(val).to_csv(os.path.join(val_dir, 'data.csv'), index=False, header=False)
    pd.DataFrame(test).to_csv(os.path.join(test_dir, 'data.csv'), index=False, header=False)

    ###############################################################################
    # Visualize the data
    logging.info('Visualize the data')
    _visualize_distribution(train, val, test, os.path.join(dataset_dir, 'visualize_distribution.pdf'))

    ###############################################################################
    # Run the analytics
    logging.info('Run the analytics')
    mean = np.mean(train[:, 0:-1], 0)
    std = np.std(train[:, 0:-1], 0)

    # Save results as CSV file in the dataset folder
    logging.info('Save results as CSV file in the dataset folder')
    df = pd.DataFrame([mean, std])
    df.index = ['mean[RGB]', 'std[RGB]']
    df.to_csv(os.path.join(dataset_dir, 'analytics.csv'), header=False)

    logging.info('Done!')
