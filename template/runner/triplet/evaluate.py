# Utils
import logging
import time

# Torch related stuff
import torch

# DeepDIVA
from util.misc import AverageMeter, accuracy


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(val_loader, model, criterion, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


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

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()
    for i, (input, target) in enumerate(data_loader):

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute output
        output = model(input_var)

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy
        acc1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(acc1[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + i)
            writer.add_scalar(logging_label + '/mb_accuracy', acc1.cpu().numpy(), epoch * len(data_loader) + i)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(data_loader) + i)
            writer.add_scalar(logging_label + '/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(data_loader) + i)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0:
            logging.info(logging_label + ' Epoch [{0}][{1}/{2}]\t'
                                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(data_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/accuracy', top1.avg, epoch)
    else:
        writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), top1.avg, epoch)

    logging.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
