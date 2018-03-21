# Utils
import logging

import numpy as np
# Torch related stuff
import torch
from torch.autograd import Variable
from tqdm import tqdm

# DeepDIVA
from template.runner.triplet.eval_metrics import ErrorRateAt95Recall


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, logging_label, no_cuda, log_interval, **kwargs):
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

    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if not no_cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), Variable(data_p, volatile=True), Variable(label)

        # Compute output
        out_a, out_p = model(data_a), model(data_p)

        # Euclidean distance
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
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
    logging.info('\33[91m ' + logging_label +' set: ErrorRate(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/ErrorRate', fpr95, epoch)
    else:
        writer.add_scalar(logging_label + '/ErrorRate_{}'.format(multi_run), fpr95, epoch)

    return fpr95
