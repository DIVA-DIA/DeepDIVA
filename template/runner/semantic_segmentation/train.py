# Utils
import logging
import time
import numpy as np

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label
from util.evaluation.metrics.accuracy import accuracy_segmentation


def train(train_loader, model, criterion, optimizer, writer, epoch, class_encodings, no_cuda=False,
          log_interval=25, **kwargs):
    """
    Training routine

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes).
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    ----------
    meanIU.avg : float
        meanIU of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None
    num_classes = len(class_encodings)

    # Instantiate the counters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        mean_iu, loss = train_one_mini_batch(model, criterion, optimizer, input, target, loss_meter, meanIU, num_classes)

        # Add loss and accuracy to Tensorboard
        log_loss = loss.item()

        if multi_run is None:
            writer.add_scalar('train/mb_loss', log_loss, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_meanIU', mean_iu, epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), log_loss, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_meanIU_{}'.format(multi_run), mean_iu, epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/meanIU', meanIU.avg, epoch)
    else:
        writer.add_scalar('train/meanIU_{}'.format(multi_run), meanIU.avg, epoch)

    logging.debug('Train epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, meanIU=meanIU))

    return meanIU.avg


def train_one_mini_batch(model, criterion, optimizer, input_var, target_var_argmax, loss_meter, meanIU_meter, num_classes):
    """
    This routing train the model passed as parameter for one mini-batch
    Parameters
    ----------
    num_classes:
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    input_var : torch.autograd.Variable
        The input data for the mini-batch
    target_var_argmax : torch.autograd.Variable
        The target data (labels) for the mini-batch
    loss_meter : AverageMeter
        Tracker for the overall loss
    meanIU_meter : AverageMeter
        Tracker for the overall meanIU
    Returns
    -------
    acc : float
        Accuracy for this mini-batch
    loss : float
        Loss for this mini-batch
    """
    # Compute output
    output = model(input_var)

    # Compute and record the loss
    loss = criterion(output, target_var_argmax)
    try:
        loss_meter.update(loss.item(), len(input_var))
    except AttributeError:
        loss_meter.update(loss.data[0], len(input_var))

    output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])
    target_argmax = target_var_argmax.data.cpu().numpy()

    # Compute and record the accuracy
    _, _, mean_iu, _ = accuracy_segmentation(target_argmax, output_argmax, num_classes)
    meanIU_meter.update(mean_iu, input_var.size(0))

    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Perform a step by updating the weights
    optimizer.step()

    # return acc, loss
    return mean_iu, loss
