# Utils
import logging
import time

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter
from util.evaluation.metrics import accuracy


def train(train_loader, model, criterion, optimizer, writer, epoch, no_cuda=False, log_interval=25,
          **kwargs):
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
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    # TODO All parts computing the accuracy are commented out. See the TODO in evaluate.py

    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, _) in pbar:

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)

        #Split the data into halves to separate the input from the GT
        satel_image, map_image = torch.chunk(input, chunks=2, dim=3)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(satel_image)
        target_var = torch.autograd.Variable(map_image)

        loss = train_one_mini_batch(model, criterion, optimizer, input_var, target_var, loss_meter, acc_meter)

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('train/mb_accuracy', acc.cpu().numpy(), epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('train/mb_accuracy_{}'.format(multi_run), acc.cpu().numpy(),
            #                   epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             # Acc1='{acc_meter.avg:.3f}\t'.format(acc_meter=acc_meter),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    # if multi_run is None:
    #     writer.add_scalar('train/accuracy', acc_meter.avg, epoch)
    # else:
    #     writer.add_scalar('train/accuracy_{}'.format(multi_run), acc_meter.avg, epoch)

    logging.debug('Train epoch[{}]: '
                  # 'Acc@1={acc_meter.avg:.3f}\t'
                  'Loss={loss.avg:.4f}\t'
                  'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, acc_meter=acc_meter))

    return acc_meter.avg


def train_one_mini_batch(model, criterion, optimizer, input_var, target_var, loss_meter, acc_meter):
    """
    This routing train the model passed as parameter for one mini-batch

    Parameters
    ----------
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    input_var : torch.autograd.Variable
        The input data for the mini-batch
    target_var : torch.autograd.Variable
        The target data (labels) for the mini-batch
    loss_meter : AverageMeter
        Tracker for the overall loss
    acc_meter : AverageMeter
        Tracker for the overall accuracy

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
    loss = criterion(output, target_var)
    loss_meter.update(loss.data[0], len(input_var))

    # Compute and record the accuracy
    # acc = accuracy(output.data, target_var.data, topk=(1,))[0]
    # acc_meter.update(acc[0], len(input_var))

    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Perform a step by updating the weights
    optimizer.step()

    # return acc, loss
    return loss