import argparse
import inspect
import os
import pickle
import sys
from multiprocessing import Pool

import matplotlib as mpl
import torch

mpl.use('Agg')

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import cv2
import numpy as np

from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA


########################################################################################################################
def tsne(features, n_components=2):
    return TSNE(n_components=n_components).fit_transform(features)


def isomap(features, n_components=2):
    return Isomap(n_components=n_components, n_jobs=-1).fit_transform(features)


def mds(features, n_components=2):
    return MDS(n_components=n_components, n_jobs=-1).fit_transform(features)


def pca(features, n_components=2):
    return PCA(n_components=n_components).fit_transform(features)


########################################################################################################################
def _make_embedding(features, labels, embedding, three_d=False):
    """
    Adapted from https://indico.io/blog/visualizing-with-t-sne/
    """

    plt.style.use(['seaborn-white', 'seaborn-paper'])
    fig = plt.figure(figsize=(8, 8))
    plt.tight_layout()
    mpl.rc("font", family="Times New Roman")

    X = features

    if three_d:
        ax = plt.axes(projection='3d')
        X_embedded = getattr(sys.modules[__name__], embedding)(X, n_components=3)
        ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=labels, cmap=plt.cm.get_cmap('jet', len(np.unique(labels))))
    else:
        X_embedded = getattr(sys.modules[__name__], embedding)(X)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', len(np.unique(labels))))

    # plt.colorbar(ticks=range(len(np.unique(labels))))

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    plt.close()
    return data


def _load_thumbnail(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (16, 16))
    return img


def _main(args):
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)

    features, preds, labels, filenames = results

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.dirname(args.output_file))
        with Pool(16) as pool:
            images = pool.map(_load_thumbnail, filenames)
        writer.add_embedding(torch.from_numpy(features), metadata=torch.from_numpy(labels),
                             # label_img=torch.from_numpy(np.array(images)).unsqueeze(1))
                             label_img=None)
        return
    viz_img = _make_embedding(features=features, labels=labels, embedding=args.embedding, three_d=args.three_d)
    cv2.imwrite(args.output_file, viz_img)
    return


if __name__ == "__main__":
    # Embedding options:
    embedding_options = [name[0] for name in inspect.getmembers(sys.modules[__name__], inspect.isfunction)]

    parser = argparse.ArgumentParser()

    parser.add_argument('--results-file',
                        type=str,
                        help='path to a results pickle file')

    parser.add_argument('--embedding',
                        help='which embedding to use for the features',
                        choices=embedding_options,
                        type=str)
    parser.add_argument('--output-file',
                        type=str,
                        default='./output.png',
                        help='path to generate output image')
    parser.add_argument('--3d',
                        dest='three_d',
                        action='store_true',
                        default=False,
                        help='enable 3d plots')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        default=False,
                        help='store embeddings to tensorboard')

    args = parser.parse_args()

    _main(args)
