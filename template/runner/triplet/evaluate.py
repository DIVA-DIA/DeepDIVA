# Utils
import logging

import numpy as np
# Torch related stuff
import torch
from torch.autograd import Variable
from tqdm import tqdm

# DeepDIVA
from template.runner.triplet.eval_metrics import ErrorRateAt95Recall
from util.misc import _prettyprint_logging_label


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate_fp95r(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate_fp95r(test_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate_fp95r(data_loader, model, criterion, writer, epoch, logging_label, no_cuda, log_interval, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    :param data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set

    :param model : torch.nn.module
        The network model being used

    :param criterion: torch.nn.loss
        The loss function used to compute the loss of the model

    :param writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    :param epoch : int
        Number of the epoch (for logging purposes)

    :param logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.

    :param no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    :param log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    :return:
        None
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    labels, distances = [], []

    multi_crop = False
    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (data_a, data_pn, label) in pbar:

        if len(data_a.size()) == 5:
            multi_crop = True

            bs, ncrops, c, h, w = data_a.size()

            data_a = data_a.view(-1, c, h, w)
            data_pn = data_pn.view(-1, c, h, w)

        if not no_cuda:
            data_a, data_pn = data_a.cuda(), data_pn.cuda()

        data_a, data_pn, label = Variable(data_a, volatile=True), Variable(data_pn, volatile=True), Variable(label)

        # Compute output
        out_a, out_pn = model(data_a), model(data_pn)

        if multi_crop:
            out_a = out_a.view(bs, ncrops, -1).mean(1)
            out_pn = out_pn.view(bs, ncrops, -1).mean(1)

        # Euclidean distance
        dists = torch.sqrt(torch.sum((out_a - out_pn) ** 2, 1))
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        # Log progress to console
        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label + ' Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))

    # Measure accuracy (FPR95)
    num_tests = len(data_loader.dataset.matches)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    distances = np.concatenate(distances, 0).reshape(num_tests)
    fpr95 = ErrorRateAt95Recall(labels, distances)
    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'FPR95={:.4f}'.format(epoch, fpr95))


    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/ErrorRate', fpr95, epoch)
    else:
        writer.add_scalar(logging_label + '/ErrorRate_{}'.format(multi_run), fpr95, epoch)

    return fpr95


def get_top_one(distances, labels):
    top_ones = []
    top_tens = []
    for i, row in enumerate(distances):
        sorted_similarity = np.argsort(row)[1:]
        gt = labels[i]
        top_one = labels[sorted_similarity[0]]
        top_ten = [labels[item] for item in sorted_similarity[:10]]
        top_ones.append([top_one, ].count(gt))
        top_tens.append((top_ten.count(gt)))
    return np.average(top_ones), np.average(top_tens)


def _evaluate_topn(data_loader, model, criterion, writer, epoch, logging_label, no_cuda, log_interval, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    :param data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set

    :param model : torch.nn.module
        The network model being used

    :param criterion: torch.nn.loss
        The loss function used to compute the loss of the model

    :param writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    :param epoch : int
        Number of the epoch (for logging purposes)

    :param logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.

    :param no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    :param log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    :return:
        None
    """
    from sklearn.metrics import pairwise_distances
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    labels, outputs = [], []

    # For use with the multi-crop transform
    multi_crop = False

    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=200)
    for batch_idx, (data, label) in pbar:

        # Check if data is provided in multi-crop form and process accordingly
        if len(data.size()) == 5:
            multi_crop = True
            bs, ncrops, c, h, w = data.size()
            data = data.view(-1, c, h, w)
        if not no_cuda:
            data = data.cuda()

        data_a, label = Variable(data, volatile=True), Variable(label)

        # Compute output
        out = model(data_a)

        if multi_crop:
            out = out.view(bs, ncrops, -1).mean(1)

        # Euclidean distance
        outputs.append(out.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        # Log progress to console
        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label + ' Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))

    # Measure accuracy (FPR95)
    num_tests = len(data_loader.dataset.data)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    outputs = np.concatenate(outputs, 0)
    distances = pairwise_distances(outputs, metric='cosine', n_jobs=16)
    top1, top10 = get_top_one(distances, labels)
    logging.info('\33[91m ' + logging_label + ' set: Top1: {}  Top10: {}\n\33[0m'.format(top1, top10))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/Top1', top1, epoch)
        writer.add_scalar(logging_label + '/Top10', top10, epoch)
    else:
        writer.add_scalar(logging_label + '/Top1{}'.format(multi_run), top1, epoch)
        writer.add_scalar(logging_label + '/Top10{}'.format(multi_run), top10, epoch)

    return top1
