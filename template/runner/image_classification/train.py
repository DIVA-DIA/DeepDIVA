# Utils
import logging
import time

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter
from util.metrics import accuracy


def train(train_loader, model, criterion, optimizer, writer, epoch, no_cuda=False, log_interval=25, **kwargs):
    """
    Training routine

    Parameters
    ----------
    :param train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.

    :param model : torch.nn.module
        The network model being used.

    :param criterion : torch.nn.loss
        The loss function used to compute the loss of the model.

    :param optimizer : torch.optim
        The optimizer used to perform the weight update.

    :param writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    :param epoch : int
        Number of the epoch (for logging purposes).

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
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy
        # acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        acc1 = accuracy(output.data, target, topk=(1,))[0]

        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run == None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_accuracy', acc1.cpu().numpy(), epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(train_loader) + batch_idx)

        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             Acc1='{top1.avg:.3f}\t'.format(top1=top1),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/accuracy', top1.avg, epoch)
    else:
        writer.add_scalar('train/accuracy_{}'.format(multi_run), top1.avg, epoch)

    logging.debug('Train epoch[{}]: '
                  'Acc@1={top1.avg:.3f}\t'
                  'Loss={loss.avg:.4f}\t'
                  'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

    return top1.avg
