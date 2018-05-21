import os
import argparse
import pickle

import numpy as np
from pandas import DataFrame
from sklearn.metrics import pairwise_distances

def _get_only_filename(path):
    return os.path.basename(path).split('.')[0]


def _main(args):
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)

    features, preds, labels, filenames = results
    distances = pairwise_distances(features, metric='cosine', n_jobs=-1)
    filenames = np.array([_get_only_filename(item) for item in filenames])

    avg_top_one = []
    results = []
    for i, row in enumerate(distances):
        sorted_similarity = np.argsort(row)[1:]
        gt = labels[i]
        top_one = labels[sorted_similarity[0]]
        avg_top_one.append([[top_one, ].count(gt)])

        tmp = []
        tmp.append(filenames[i])
        query_results = filenames[sorted_similarity[:args.num_results]]
        _ = [tmp.append(item) for item in query_results]
        results.append(tmp)

    tmp = []
    tmp.append('Query')
    if args.num_results == None:
        args.num_results = len(features) - 1
    _ = [tmp.append('R{}'.format(i + 1)) for i in range(args.num_results)]

    dframe = DataFrame(results, columns=tmp)
    dframe.to_csv(args.output_file, index=False)
    print('Precision@1: {}'.format(np.average(avg_top_one)))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--results-file',
                        type=str,
                        help='path to a results pickle file')

    parser.add_argument('--output-file',
                        type=str,
                        default='./output.png',
                        help='path to generate output CSV')
    parser.add_argument('--num-results',
                        type=int,
                        default=None,
                        help='save top N results for each query')

    args = parser.parse_args()

    _main(args)
