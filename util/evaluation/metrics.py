"""
This contains several commonly used metrics.

"""
import datetime
import logging
import time
from multiprocessing import Pool

import numpy as np


def _apk(query, predicted, k='full'):
    """
    Computes the average precision@k.

    Parameters
    ----------
    query : int
        Query label.
    predicted : List(int)
        Ordered list where each element is a label.
    k : str or int
        If int, cutoff for retrieval is set to K
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.
                For e.g.,
                    query = 0
                    predicted = [0, 0, 1, 1, 0]
                    if k == 'full', then k is set to 5
                    if k == 'auto', then k is set to num of 'query' values in 'predicted',
                    i.e., k=3 as there as 3 of them in 'predicted'

    Returns
    -------
    average_prec : float
        Average Precision@k

    """
    if k == 'auto':
        k = predicted.count(query)
    elif k == 'full':
        k = len(predicted)

    if k == 0 or len(predicted) == 0:
        return 0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0  # The score is the precision@i integrated over i=1:k
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == query:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / k

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
    # Crop the predicted list.
    predicted = np.array(predicted[:min(k, len(predicted))])

    # Make an empty array for relevant queries.
    relv = np.zeros(len(predicted))

    # Find all locations where the predicted value matches the query and vice-versa.
    hit_locs = np.where(predicted == query)[0]
    non_hit_locs = np.where(predicted != query)[0]

    # Set all `hit_locs` to be 1. [0,0,0,0,0,0] -> [0,1,0,1,0,1]
    relv[hit_locs] = 1
    # Compute the sum of all elements till the particular element. [0,1,0,1,0,1] -> [0,1,1,2,2,3]
    relv = np.cumsum(relv)
    #  Set all `non_hit_locs` to be zero. [0,1,1,2,2,3] -> [0,1,0,2,0,3]
    relv[non_hit_locs] = 0
    # Divide element-wise by [0/1,1/2,0/3,2/4,0/5,3/6] and sum the array.
    score = np.sum(np.divide(relv, np.arange(1, relv.shape[0] + 1)))

    average_prec = score / k

    return average_prec


def _apk_old(query, predicted, k='full'):
    """Computes the average precision@K.

    Parameters
    ----------
    query : int
        Query label.
    predicted : list of int
        Where each int is a label.
    k : str or int
        If int, cutoff for retrieval is set to K
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.
                For e.g.,
                    query = 0
                    predicted = [0, 0, 1, 1, 0]
                    if k == 'full', then k is set to 5
                    if k == 'auto', then k is set to num of 'query' values in 'predicted',
                    i.e., k=3 as there as 3 of them in 'predicted'

    Returns
    -------
    average_prec : float
        Average Precision@K

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
    # Crop the predicted list.
    predicted = np.array(predicted[:min(k, len(predicted))])

    # Make an empty array for relevant queries.
    relv = np.zeros(len(predicted))

    # Find all locations where the predicted value matches the query and vice-versa.
    hit_locs = np.where(predicted == query)[0]
    non_hit_locs = np.where(predicted != query)[0]

    # Set all `hit_locs` to be 1. [0,0,0,0,0,0] -> [0,1,0,1,0,1]
    relv[hit_locs] = 1
    # Compute the sum of all elements till the particular element. [0,1,0,1,0,1] -> [0,1,1,2,2,3]
    relv = np.cumsum(relv)
    #  Set all `non_hit_locs` to be zero. [0,1,1,2,2,3] -> [0,1,0,2,0,3]
    relv[non_hit_locs] = 0
    # Divide element-wise by [0/1,1/2,0/3,2/4,0/5,3/6] and sum the array.
    score = np.sum(np.divide(relv, np.arange(1, relv.shape[0] + 1)))

    average_prec = score / num_relevant

    return average_prec


def _mapk(query, predicted, k=None, workers=1):
    """Compute the mean Average Precision@K.

    Parameters
    ----------
    query : list
        List of queries.
    predicted : list of list
        Predicted responses for each query.
    k : str or int
        If int, cutoff for retrieval is set to `k`
        If str, 'full' means cutoff is til the end of predicted
                'auto' means cutoff is set to number of relevant queries.
                For e.g.,
                    `query` = 0
                    `predicted` = [0, 0, 1, 1, 0]
                    if `k` == 'full', then `k` is set to 5
                    if `k` == 'auto', then `k` is set to num of `query` values in `predicted`,
                    i.e., `k`=3 as there as 3 of them in `predicted`.
    workers : int
        Number of parallel workers used to compute the AP@k

    Returns
    -------
    map_score : float
        The mean average precision@K.

    """
    if workers == 1:
        return np.mean([_apk(q, p, k) for q, p in zip(query, predicted)])
    with Pool(workers) as pool:
        vals = [[q, p, k] for q, p in zip(query, predicted)]
        aps = pool.starmap(_apk, vals)
    map_score = np.mean(aps)
    return map_score


def compute_mapk(distances, labels, k, workers=None):
    """Convenience function to convert a grid of pairwise distances to predicted
    elements, to evaluate mean average precision (at K).

    Parameters
    ----------
    distances : ndarray
        A numpy array containing pairwise distances between all elements
    labels : list
        Ground truth labels for every element
    k : int
        Maximum number of predicted elements

    Returns
    -------
    map_score : float
        The mean average precision@K.
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


def accuracy(predicted, target, topk=(1,)):
    """Computes the accuracy@K for the specified values of K

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    predicted : torch.FloatTensor
        The predicted output values of the model.
    target : torch.LongTensor
        The ground truth for the corresponding output.
    topk : tuple
        Multiple values for K can be specified in a tuple, and the
        different accuracies@K will be computed.

    Returns
    -------
    res : list
        List of accuracies computed at the different K's specified in `topk`

    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = predicted.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
