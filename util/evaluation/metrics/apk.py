# Utils
import datetime
import logging
import time
import numpy as np


def apk(query, predicted, k='full'):
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

        Example:
            query = 0
            predicted = [0, 0, 1, 1, 0]
            if k == 'full', then k is set to 5
            if k == 'auto', then k is set to num of 'query' values in 'predicted',
            i.e., k=3 as there as 3 of them in 'predicted'

    Returns
    -------
    float
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

    predicted = np.array(predicted)

    # Non-vectorized version.
    # score = 0.0  # The score is the precision@i integrated over i=1:k
    # num_hits = 0.0
    #
    # for i, p in enumerate(predicted):
    #     if p == query:
    #         num_hits += 1.0
    #         score += num_hits / (i + 1.0)
    #
    # return score / k

    # Make an empty array for relevant queries.
    relevant = np.zeros(len(predicted))

    # Find all locations where the predicted value matches the query and vice-versa.
    hit_locs = np.where(predicted == query)[0]
    non_hit_locs = np.where(predicted != query)[0]

    # Set all `hit_locs` to be 1. [0,0,0,0,0,0] -> [0,1,0,1,0,1]
    relevant[hit_locs] = 1
    # Compute the sum of all elements till the particular element. [0,1,0,1,0,1] -> [0,1,1,2,2,3]
    relevant = np.cumsum(relevant)
    #  Set all `non_hit_locs` to be zero. [0,1,1,2,2,3] -> [0,1,0,2,0,3]
    relevant[non_hit_locs] = 0
    # Divide element-wise by [0/1,1/2,0/3,2/4,0/5,3/6] and sum the array.
    score = np.sum(np.divide(relevant, np.arange(1, relevant.shape[0] + 1)))

    return score / k


def mapk(query, predicted, k=None, workers=1):
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
    float
        The mean average precision@K.
    dict{label, float}
        The per class mean averages precision @k
    """
    results = np.array([apk(q, p, k) for q, p in zip(query, predicted)])
    per_class_mapk = {str(l): np.mean(np.array(results)[np.where(query == l)[0]]) for l in np.unique(query)}
    return np.mean(results), per_class_mapk
    # The overhead of the pool is killing any possible speedup.
    # In order to make this parallel (if ever needed) one should create a Process class which swallows
    # 1/`workers` part of `vals`, such that only `workers` threads are created.

    # if workers == 1:
    #     return np.mean([_apk(q, p, k) for q, p in zip(query, predicted)])
    # with Pool(workers) as pool:
    #     vals = [[q, p, k] for q, p in zip(query, predicted)]
    #     aps = pool.starmap(_apk, vals)
    #     return np.mean(aps)


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
    float
        The mean average precision@K.
    dict{label, float}
        The per class mean averages precision @k
    """

    # Resolve k
    k = k if k == 'auto' or k == 'full' else int(k)

    # Reduce the size of distances that would anyway not be used afterwards. This makes sorting them faster.
    max_count = k
    if k == 'full':
        max_count = len(labels)
    if k == 'auto':
        # Take the highest frequency in the labels i.e. the highest possible 'auto' value for all entries
        max_count = np.max(np.unique(labels, return_counts=True)[1])

    # Fetch the index of the lowest `max_count` (k) elements
    t = time.time()
    ind = np.argpartition(distances, max_count)[:, :max_count]
    # Find the sorting sequence according to the shortest distances selected from `ind`
    ssd = np.argsort(np.array(distances)[np.arange(distances.shape[0])[:, None], ind], axis=1)
    # Consequently sort `ind`
    ind = ind[np.arange(ind.shape[0])[:, None], ssd]
    # Now `ind` contains the sorted indexes of the lowest `max_count` (k) elements
    # Resolve the labels of the elements referred by `ind`
    sorted_predictions = [list(labels[row][1:]) for row in ind]
    logging.debug('Finished computing sorted predictions in {} seconds'
                  .format(datetime.timedelta(seconds=int(time.time() - t))))

    if workers is None:
        workers = 16 if k == 'auto' or k == 'full' else 1

    return mapk(labels, sorted_predictions, k, workers)
