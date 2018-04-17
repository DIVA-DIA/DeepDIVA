# Utils
import datetime
import logging
import time

import numpy as np
# Torch related stuff
from torch.autograd import Variable
from tqdm import tqdm

# DeepDIVA
from util.metrics import compute_mapk


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate_map(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate_map(test_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate_map(data_loader, model, criterion, writer, epoch, logging_label, no_cuda, log_interval, map, **kwargs):
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

    :param map : str
        Specify value for mAP computation. Possible values are ("auto", "full" or specify K for AP@K)

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
    num_tests = len(data_loader.dataset.file_names)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    outputs = np.concatenate(outputs, 0)
    distances = pairwise_distances(outputs, metric='cosine', n_jobs=16)
    logging.debug('Computed pairwise distances')
    logging.debug('Distance matrix shape: {}'.format(distances.shape))
    t = time.time()
    mAP_score = compute_mapk(distances, labels, k=map)
    logging.debug('Completed evaluation of mAP in {}'.format(datetime.timedelta(seconds=int(time.time() - t))))

    logging.info('\33[91m ' + logging_label + ' set: mAP: {}\n\33[0m'.format(mAP_score))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/mAP', mAP_score, epoch)
    else:
        writer.add_scalar(logging_label + '/mAP{}'.format(multi_run), mAP_score, epoch)

    return mAP_score
