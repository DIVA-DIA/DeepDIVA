# Utils
import time

# Torch related stuff
from torch.autograd import Variable
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter


def train(train_loader, model, criterion, optimizer, writer, epoch, no_cuda, log_interval=25, **kwargs):
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
    int
        Placeholder 0. In the future this should become the FPR95
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if len(data_a.size()) == 5:
            bs, ncrops, c, h, w = data_a.size()

            data_a = data_a.view(-1, c, h, w)
            data_p = data_p.view(-1, c, h, w)
            data_n = data_n.view(-1, c, h, w)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            data_a, data_p, data_n = data_a.cuda(async=True), data_p.cuda(async=True), data_n.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)

        # Compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        if len(data_a.size()) == 5:
            out_a = out_a.view(bs, ncrops, -1).mean(1)
            out_p = out_p.view(bs, ncrops, -1).mean(1)
            out_n = out_n.view(bs, ncrops, -1).mean(1)

        # Compute and record the loss
        loss = criterion(out_p, out_a, out_n)

        losses.update(loss.data[0], data_a.size(0))

        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a),
                    len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    losses.avg))

        # Add mb loss to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0], epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return 0
