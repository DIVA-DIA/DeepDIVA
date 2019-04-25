# Utils
import logging
import time
import warnings

import numpy as np
# Torch related stuff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from util.evaluation.metrics import accuracy
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard
from util.visualization.confusion_matrix_heatmap import make_heatmap


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, logging_label, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)

    with torch.no_grad():
        for batch_idx, (input, target) in pbar:

            # Measure data loading time
            data_time.update(time.time() - end)

            # Moving data to GPU
            if not no_cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # Compute output
            output = model(input)

            # Compute and record the loss
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # Apply sigmoid and take everything above a threshold of 0.5
            squashed_output = torch.nn.Sigmoid()(output).data.cpu().numpy()
            target_vals = target.cpu().numpy().astype(np.int)

            # jss = compute_jss(target_vals, get_preds_from_minibatch(squashed_output))
            # top1.update(jss, input.size(0))

            # Store results of each minibatch
            _ = [preds.append(item) for item in get_preds_from_minibatch(squashed_output)]
            _ = [targets.append(item) for item in target.cpu().numpy()]

            # Add loss and accuracy to Tensorboard
            if multi_run is None:
                writer.add_scalar(logging_label + '/mb_loss', loss.item(), epoch * len(data_loader) + batch_idx)
                # writer.add_scalar(logging_label + '/mb_jaccard_similarity', jss, epoch * len(data_loader) + batch_idx)
            else:
                writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.item(),
                                  epoch * len(data_loader) + batch_idx)
                # writer.add_scalar(logging_label + '/mb_jaccard_similarity_{}'.format(multi_run), jss,
                #                   epoch * len(data_loader) + batch_idx)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                pbar.set_description(logging_label +
                                     ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

                pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                                 Loss='{loss.avg:.4f}\t'.format(loss=losses),
                                 # JSS='{top1.avg:.3f}\t'.format(top1=top1),
                                 Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Generate a classification report for each epoch
    targets = np.array(targets).astype(np.int)
    preds = np.array(preds).astype(np.int)
    _log_classification_report(data_loader, epoch, preds, targets, writer)
    jss_epoch = compute_jss(targets, preds)
    # try:
    #     np.testing.assert_approx_equal(jss_epoch, top1.avg)
    # except:
    #     logging.error('Computed JSS scores do not match')
    #     logging.error('JSS: {} Avg: {}'.format(jss_epoch, top1.avg))

    # # Logging the epoch-wise JSS
    if multi_run is None:
        writer.add_scalar(logging_label + '/loss', losses.avg, epoch)
        writer.add_scalar(logging_label + '/jaccard_similarity', jss_epoch, epoch)
    else:
        writer.add_scalar(logging_label + '/loss_{}'.format(multi_run), losses.avg, epoch)
        writer.add_scalar(logging_label + '/jaccard_similarity_{}'.format(multi_run), jss_epoch, epoch)

    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'JSS={jss_epoch:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, jss_epoch=jss_epoch))


    return jss_epoch


def get_preds_from_minibatch(minibatch):
    preds = []
    for row in minibatch:
        tmp = [1 if item > 0.5 else 0 for item in row]
        preds.append(tmp)
    preds = np.array(preds).astype(np.int)
    return preds


def compute_jss(target, preds):
    score = 0
    num_classes = len(target[0])
    for i in range(num_classes):
        score += jaccard_similarity_score(target[:,i], preds[:,i])

    score = score/num_classes
    return score


def jaccard_similarity_score(targets, preds):
    assert len(targets) == len(preds)
    assert len(targets.shape) == 1
    assert len(preds.shape) == 1

    locs_targets = set(np.where(targets == 1)[0])
    locs_preds = set(np.where(preds == 1)[0])

    try:
        score = len(locs_targets.intersection(locs_preds)) / len(locs_targets.union(locs_preds))
    except:
        print('Exception!')

    return score


def _log_classification_report(data_loader, epoch, preds, targets, writer):
    """
    This routine computes and prints on Tensorboard TEXT a classification
    report with F1 score, Precision, Recall and similar metrics computed
    per-class.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    epoch : int
        Number of the epoch (for logging purposes)
    preds : list
        List of all predictions of the model for this epoch
    targets : list
        List of all correct labels for this epoch
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    Returns
    -------
        None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classification_report_string = str(classification_report(y_true=targets,
                                                                 y_pred=preds,
                                                                 target_names=[str(item) for item in
                                                                               data_loader.dataset.classes]))
    # Fix for TB writer. Its an ugly workaround to have it printed nicely in the TEXT section of TB.
    classification_report_string = classification_report_string.replace('\n ', '\n\n       ')
    classification_report_string = classification_report_string.replace('precision', '      precision', 1)
    classification_report_string = classification_report_string.replace('avg', '      avg', 1)

    writer.add_text('Classification Report for epoch {}\n'.format(epoch), '\n' + classification_report_string, epoch)
