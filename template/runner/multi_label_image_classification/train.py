# Utils
import logging
import time
import numpy as np
from tqdm import tqdm
# from sklearn.metrics import jaccard_similarity_score
# Torch related stuff
import torch

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
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    jss_meter = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

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

        jss, loss, target_vals, pred_vals = train_one_mini_batch(model, criterion, optimizer, input, target,
                                                                 loss_meter, jss_meter)

        # Store results of each minibatch
        _ = [preds.append(item) for item in pred_vals]
        _ = [targets.append(item) for item in target_vals]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.item(), epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('train/mb_jaccard_similarity', jss, epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.item(),
                              epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('train/mb_jaccard_similarity_{}'.format(multi_run), jss,
            #                   epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             # JSS='{jss_meter.avg:.3f}\t'.format(jss_meter=jss_meter),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Generate the epoch wise JSS
    targets = np.array(targets).astype(np.int)
    preds = np.array(preds).astype(np.int)
    jss_epoch = compute_jss(targets, preds)
    # try:
    #     np.testing.assert_approx_equal(jss_epoch, jss_meter.avg)
    # except:
    #     logging.error('Computed JSS scores do not match')
    #     logging.error('JSS: {} Avg: {}'.format(jss_epoch, jss_meter.avg))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/loss', loss_meter.avg, epoch)
        writer.add_scalar('train/jaccard_similarity', jss_epoch, epoch)
    else:
        writer.add_scalar('train/loss_{}'.format(multi_run), loss_meter.avg, epoch)
        writer.add_scalar('train/jaccard_similarity_{}'.format(multi_run), jss_epoch, epoch)

    logging.debug('Train epoch[{}]: '
                  'JSS={jss_epoch:.3f}\t'
                  'Loss={loss.avg:.4f}\t'
                  'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, jss_epoch=jss_epoch))

    return jss_epoch


def train_one_mini_batch(model, criterion, optimizer, input, target, loss_meter, jss_meter):
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
    input : torch.autograd.Variable
        The input data for the mini-batch
    target : torch.autograd.Variable
        The target data (labels) for the mini-batch
    loss_meter : AverageMeter
        Tracker for the overall loss
    jss_meter : AverageMeter
        Tracker for the overall Jaccard Similarity Score


    Returns
    -------
    loss : float
        Loss for this mini-batch
    """
    # Compute output
    output = model(input)

    # Compute and record the loss
    loss = criterion(output, target)
    loss_meter.update(loss.item(), len(input))

    # Apply sigmoid and take everything above a threshold of 0.5
    squashed_output = torch.nn.Sigmoid()(output).data.cpu().numpy()
    preds = get_preds_from_minibatch(squashed_output)
    target_vals = target.data.cpu().numpy().astype(np.int)

    # # Compute and record the Jaccard Similarity Score
    # jss = compute_jss(target_vals, preds)
    # jss_meter.update(jss, len(input))
    jss = None
    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Perform a step by updating the weights
    optimizer.step()

    return jss, loss, target_vals, preds


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
        score += jaccard_similarity_score(target[:, i], preds[:, i])

    score = score / num_classes
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