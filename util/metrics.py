import time
import pickle
import logging
import argparse
import datetime
from multiprocessing import Pool

import numpy as np


def _apk(query, predicted, k='full'):
    """
    Computes the average precision at K.
    Parameters
    ----------
    query: int
        Query label
    predicted: list
        List of ints, where each int is a label.
    k: str or int
        If int, cutoff for retrieval is set to K
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.
                For e.g.,
                    q = 0
                    predicted = [0,0,1,1]
                    if k == 'full', then k is set to 4
                    if k == 'auto', then k is set to num of predicted values
                        that equal query, i.e., k will be 2

    Returns
    -------
    average_prec: float
        Average Precision at K

    """

    num_relevant = predicted.count(query)

    if k == 'auto':
        k = num_relevant
    elif k == 'full':
        k = len(predicted)

    if num_relevant == 0:
        return np.nan

    # Non-vectorized version.
    # score = 0.0
    # hit = 0
    # for i in range(min(k, len(predicted))):
    #     prec = predicted[:i + 1].count(query) / (i + 1)
    #     relv = 1 if predicted[i] == query else 0
    #     score += prec * relv
    #
    #     hit += relv
    #     if hit >= num_relevant:
    #         break

    # Vectorized form
    predicted = np.array(predicted[:min(k, len(predicted))])

    relv = np.zeros(len(predicted))

    hit_locs = np.where(predicted == query)[0]
    non_hit_locs = np.where(predicted != query)[0]

    relv[hit_locs] = 1
    relv = np.cumsum(relv)
    relv[non_hit_locs] = 0

    score = np.sum(np.divide(relv, np.arange(1, relv.shape[0] + 1)))

    average_prec = score / num_relevant

    return average_prec


def _mapk(query, predicted, k=None, workers=1):
    if workers == 1:
        return np.mean([_apk(q, p, k) for q, p in zip(query, predicted)])
    with Pool(workers) as pool:
        vals = [[q, p, k] for q, p in zip(query, predicted)]
        aps = pool.starmap(_apk, vals)
    return np.mean(aps)


def compute_mapk(distances, labels, k, workers=None):
    """
    Convenience function to convert a grid of pairwise distances to predicted
    elements, to evaluate mAP(at K).

    Parameters
    ----------
    distances : ndarray
                A numpy array containing pairwise distances between all elements
    labels  :   list
                Ground truth labels for every element
    k   :   int
                Maximum number of predicted elements

    Returns
    -------
    score   :   double
                The mean average precision at K over the input lists
    """
    k = k if k == 'auto' or k == 'full' else int(k)

    if workers == None:
        workers = 16 if k == 'auto' or k == 'full' else 1

    t = time.time()
    sorted_predictions = [list(labels[np.argsort(dist_row)][1:]) for dist_row in distances]
    logging.debug('Finished sorting distance matrix in {} seconds'
                  .format(datetime.timedelta(seconds=int(time.time() - t))))

    queries = labels

    t = time.time()
    map_score = _mapk(queries, sorted_predictions, k, workers)
    logging.debug('Completed evaluation of mAP in {}'.format(datetime.timedelta(seconds=int(time.time() - t))))

    return map_score


def main(args):
    with open(args.file, 'rb') as f:
        dist, labels = pickle.load(f)
    t = time.time()
    print(compute_mapk(dist, labels, k=args.cutoff, workers=args.workers))
    print('Time taken: {}'.format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('file',
                        type=str,
                        help='path to pickle file containing distances and labels')
    parser.add_argument('-j',
                        '--workers',
                        default=1,
                        type=int,
                        help='number of workers')
    parser.add_argument('-k',
                        '--cutoff',
                        default=None,
                        type=int,
                        help='K in the AP@K')

    args = parser.parse_args()
    main(args)
