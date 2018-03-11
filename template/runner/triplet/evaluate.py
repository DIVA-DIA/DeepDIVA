# Utils
import numpy as np
# Torch related stuff
import torch
# DeepDIVA
from torch.autograd import Variable
from tqdm import tqdm

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

    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(data_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if not no_cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))

    # measure accuracy (FPR95)
    num_tests = data_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, distances)
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    # Triplet.logger.log_value('fpr95', fpr95)

    #############################DD
    #
    # # Instantiate the counters
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    #
    # # Switch to evaluate mode (turn off dropout & such )
    # model.eval()
    #
    # # Iterate over whole evaluation set
    # end = time.time()
    # for i, (input, target) in enumerate(data_loader):
    #
    #     # Moving data to GPU
    #     if not no_cuda:
    #         input = input.cuda(async=True)
    #         target = target.cuda(async=True)
    #
    #     # Convert the input and its labels to Torch Variables
    #     input_var = torch.autograd.Variable(input, volatile=True)
    #     target_var = torch.autograd.Variable(target, volatile=True)
    #
    #     # Compute output
    #     output = model(input_var)
    #
    #     # Compute and record the loss
    #     loss = criterion(output, target_var)
    #     losses.update(loss.data[0], input.size(0))
    #
    #     # Compute and record the accuracy
    #     acc1 = accuracy(output.data, target, topk=(1,))[0]
    #     top1.update(acc1[0], input.size(0))
    #
    #     # Add loss and accuracy to Tensorboard
    #     if multi_run is None:
    #         writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + i)
    #         writer.add_scalar(logging_label + '/mb_accuracy', acc1.cpu().numpy(), epoch * len(data_loader) + i)
    #     else:
    #         writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
    #                           epoch * len(data_loader) + i)
    #         writer.add_scalar(logging_label + '/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
    #                           epoch * len(data_loader) + i)
    #
    #     # Measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()
    #
    #     if i % log_interval == 0:
    #         logging.info(logging_label + ' Epoch [{0}][{1}/{2}]\t'
    #                                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
    #             epoch, i, len(data_loader), batch_time=batch_time, loss=losses,
    #             top1=top1))
    #
    # # Logging the epoch-wise accuracy
    # if multi_run is None:
    #     writer.add_scalar(logging_label + '/accuracy', top1.avg, epoch)
    # else:
    #     writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), top1.avg, epoch)
    #
    # logging.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    #
    # return top1.avg
