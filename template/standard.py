"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

# Utils
import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback

# Tensor board
import tensorboardX
# Torch related stuff
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# DeepDIVA
import dataset
import model as models
from init.initializer import *
from util.misc import AverageMeter, accuracy
from util.visualization.mean_std_plot import plot_mean_variance


def main(args, writer, log_folder, **kwargs):
    """
    This is the main routine where train() and validate() are called.
    :return:
        None
    """

    model_expected_input_size = models.__dict__[args.model]().expected_input_size
    logging.info('Model {} expects input size of {}'.format(args.model,
                                                            model_expected_input_size))

    # Setting up dataset and dataloaders
    train_loader, val_loader, test_loader, num_classes = \
        set_up_dataloaders(model_expected_input_size=model_expected_input_size,
                           args=args)

    # Setting up model, optimizer, criterion
    model, criterion, optimizer, best_prec = set_up_model(num_classes=num_classes,
                                                          args=args)

    val_precs = np.zeros((args.epochs - args.start_epoch))
    train_precs = np.zeros((args.epochs - args.start_epoch))

    # Begin training
    logging.info('Begin training')
    for i in range(args.start_epoch, args.epochs):
        val_precs[i] = validate(val_loader, model, criterion, writer, i, **kwargs)
        train_precs[i] = train(train_loader, model, criterion, optimizer, writer, i, **kwargs)
        if args.decay_lr is not None:
            adjust_learning_rate(optimizer, i, args.decay_lr)
        maybe_save_model(i, val_precs[i], best_prec, model, optimizer, log_folder)

    test_prec = test(test_loader, model, criterion, writer, i, **kwargs)
    logging.info('Training completed')

    return train_precs, val_precs, test_prec


def maybe_save_model(i, val_prec1, best_prec1, model, optimizer, log_folder):
    is_best = val_prec1 > best_prec1
    best_prec1 = max(val_prec1, best_prec1)
    save_checkpoint({
        'epoch': i + 1,
        'arch': str(type(model)),
        'state_dict': model.state_dict(),
        'best_prec': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filename=os.path.join(log_folder, 'checkpoint.pth.tar'))


#######################################################################################################################

def set_up_model(num_classes, args):
    # Initialize the model
    logging.info('Setting up model {}'.format(args.model))
    model = models.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    # Init the model
    # if args.init:
    #    init_model(model=model, data_loader=train_loader, num_points=50000)

    # Transfer model to GPU (if desired)
    if not args.no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            val_losses = []
            val_losses.append(checkpoint['val_loss'])
            logging.info("Loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logging.warning("No checkpoint found at '{}'".format(args.resume))
    else:
        best_prec = 0.0
    return model, criterion, optimizer, best_prec


#######################################################################################################################

def set_up_dataloaders(model_expected_input_size, args):
    # Set up datasets and dataloaders
    # Initialize train,val and test datasets.
    logging.info('Loading datasets')
    train_ds = dataset.__dict__[args.dataset](root='.data/',
                                              train=True,
                                              download=True)

    val_ds = dataset.__dict__[args.dataset](root='.data/',
                                            train=False,
                                            val=True,
                                            download=True)

    test_ds = dataset.__dict__[args.dataset](root='.data/',
                                             train=False,
                                             download=True)

    # Set up dataset transforms
    train_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    val_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    test_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    # Setup dataloaders
    logging.info('Setting up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.num_classes


#######################################################################################################################

def train(train_loader, model, criterion, optimizer, writer, epoch):
    """
    Training routine
    :param train_loader:    torch.utils.data.DataLoader
        The dataloader of the train set
    :param model:           torch.nn.module
        The network model being used
    :param criterion:       torch.nn.loss
        The loss function used to compute the loss of the model
    :param optimizer:       torch.optim
        The optimizer used to perform the weight update
    :param epoch:
        Number of the epoch (mainly for logging purposes)
    :return:
        None
    """
    multi_run = kwargs['multi_run'] if 'multi_run' in kwargs else None

    # Init the counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not args.no_cuda:
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
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run == None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + i)
            writer.add_scalar('train/mb_accuracy', acc1.cpu().numpy(), epoch * len(train_loader) + i)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(train_loader) + i)
            writer.add_scalar('train/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(train_loader) + i)

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
        if i % args.log_interval == 0:
            logging.info('Epoch [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    # Logging the epoch-wise accuracy
    if multi_run == None:
        writer.add_scalar('train/accuracy', top1.avg, epoch)
    else:
        writer.add_scalar('train/accuracy_{}'.format(multi_run), top1.avg, epoch)

    return top1.avg


def validate(val_loader, model, criterion, writer, epoch, **kwargs):
    """
    The validation routine
    :param val_loader:    torch.utils.data.DataLoader
        The dataloader of the train set
    :param model:           torch.nn.module
        The network model being used
    :param criterion:       torch.nn.loss
        The loss function used to compute the loss of the model
    :param epoch:
        Number of the epoch (mainly for logging purposes)
    :return:
        None
    """
    multi_run = kwargs['multi_run'] if 'multi_run' in kwargs else None

    # Init the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole validation set
    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        # Moving data to GPU
        if not args.no_cuda:
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
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run == None:
            writer.add_scalar('val/mb_loss', loss.data[0], epoch * len(val_loader) + i)
            writer.add_scalar('val/mb_accuracy', acc1.cpu().numpy(), epoch * len(val_loader) + i)
        else:
            writer.add_scalar('val/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(val_loader) + i)
            writer.add_scalar('val/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(val_loader) + i)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            logging.info('Epoch [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # Logging the epoch-wise accuracy
    if multi_run == None:
        writer.add_scalar('val/accuracy', top1.avg, epoch)
    else:
        writer.add_scalar('val/accuracy_{}'.format(multi_run), top1.avg, epoch)

    logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    return top1.avg


def test(val_loader, model, criterion, writer, epoch, **kwargs):
    """
    The validation routine
    :param val_loader:    torch.utils.data.DataLoader
        The dataloader of the train set
    :param model:           torch.nn.module
        The network model being used
    :param criterion:       torch.nn.loss
        The loss function used to compute the loss of the model
    :param epoch:
        Number of the epoch (mainly for logging purposes)
    :return:
        None
    """
    multi_run = kwargs['multi_run'] if 'multi_run' in kwargs else None

    # Init the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole validation set
    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        # Moving data to GPU
        if not args.no_cuda:
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
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        if multi_run == None:
            writer.add_scalar('test/mb_loss', loss.data[0], epoch * len(val_loader) + i)
            writer.add_scalar('test/mb_accuracy', acc1.cpu().numpy(), epoch * len(val_loader) + i)
        else:
            writer.add_scalar('test/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(val_loader) + i)
            writer.add_scalar('test/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(val_loader) + i)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            logging.info('Test Epoch [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # Logging the epoch-wise accuracy
    if multi_run == None:
        writer.add_scalar('test/accuracy', top1.avg, epoch - 1)
    else:
        writer.add_scalar('test/accuracy_{}'.format(multi_run), top1.avg, epoch - 1)

    logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, epoch, num_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    lr = args.lr * (0.1 ** (epoch // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))


def set_up_logging(args):
    # Experiment name override
    if args.experiment_name is None:
        vars(args)['experiment_name'] = input("Experiment name:")

    # Setup Logging
    basename = args.log_dir
    experiment_name = args.experiment_name
    if not args.log_folder:
        log_folder = os.path.join(basename,
                                  experiment_name,
                                  args.dataset,
                                  args.model,
                                  args.optimizer,
                                  str(args.lr),
                                  '{}'.format(time.strftime('%y-%m-%d-%Hh-%Mm-%Ss')))
    else:
        log_folder = args.log_folder
    logfile = 'logs.txt'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        filename=os.path.join(log_folder, logfile),
        level=logging.INFO)
    logging.info(
        'Set up logging. Log file: {}'.format(os.path.join(log_folder, logfile)))

    # Save args to logs_folder
    logging.info(
        'Arguments saved to: {}'.format(os.path.join(log_folder, 'args.txt')))
    with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Set up logging to console
    if not args.quiet:
        fmtr = logging.Formatter(fmt='%(funcName)s %(levelname)s: %(message)s')
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = fmtr
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(stderr_handler)
        logging.info('Printing activity to the console')

    return log_folder


def set_up_env(args):
    # Set visible GPUs
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Seed the random

    if args.seed:
        try:
            assert args.multi_run == None
        except:
            logging.warning('Arguments for seed AND multi-run should not be active at the same time!')
            raise SystemExit
        if args.workers > 1:
            logging.warning('Setting seed when workers > 1 may lead to non-deterministic outcomes!')
        # Python
        random.seed(args.seed)

        # Numpy random
        np.random.seed(args.seed)

        # Torch random
        torch.manual_seed(args.seed)
        if not args.no_cuda:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.enabled = False
    return


if __name__ == "__main__":

    model_options = [name for name in models.__dict__ if callable(models.__dict__[name])]
    dataset_options = [name for name in dataset.__dict__ if callable(dataset.__dict__[name])]
    optimizer_options = [name for name in torch.optim.__dict__ if callable(torch.optim.__dict__[name])]

    ###############################################################################
    # Argument Parser

    # Training Settings
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Template for training a network on a dataset')

    parser_general = parser.add_argument_group('GENERAL', 'General Options')
    parser_data = parser.add_argument_group('DATA', 'Dataset Options')
    parser_train = parser.add_argument_group('TRAIN', 'Training Options')
    parser_system = parser.add_argument_group('SYS', 'System Options')

    # General Options
    parser_general.add_argument('--experiment-name',
                                help='provide a meaningful and descriptive name to this run',
                                default=None, type=str)
    parser_general.add_argument('--quiet',
                                action='store_true',
                                help='Do not print to stdout (log only).')
    parser_general.add_argument('--multi-run',
                                type=int,
                                default=None, help='run main N times with different random seeds')

    # Data Options
    parser_data.add_argument('--dataset',
                             choices=dataset_options,
                             help='which dataset to train/test on', default='CIFAR10')
    parser_data.add_argument('--log-dir',
                             help='where to save logs', default='./data/')
    parser_data.add_argument('--log-folder',
                             help='override default log folder (to resume logging of experiment)',
                             default=None,
                             type=str)

    # Training Options
    parser_train.add_argument('--model',
                              choices=model_options,
                              help='which model to use for training',
                              type=str, default='CNN_Basic')
    parser_train.add_argument('--lr',
                              help='learning rate to be used for training',
                              type=float, default=0.001)
    parser_train.add_argument('--optimizer',
                              choices=optimizer_options,
                              help='optimizer to be used for training',
                              default='Adam')
    parser_train.add_argument('--batch-size',
                              help='input batch size for training',
                              type=int, default=64)
    parser_train.add_argument('--epochs',
                              help='how many epochs to train',
                              type=int, default=20)
    parser_train.add_argument('--resume',
                              help='path to latest checkpoint',
                              default=None, type=str)
    parser_train.add_argument('--pretrained',
                              default=False, action='store_true',
                              help='use pretrained model. (Not applicable for all models)')
    parser_train.add_argument('--decay_lr',
                              default=None, type=int, help='drop LR by 10 every N epochs')
    parser_train.add_argument('--start-epoch', default=0, type=int, metavar='N',
                              help='manual epoch number (useful on restarts)')
    # System Options

    parser_system.add_argument('--gpu-id',
                               default=None,
                               help='which GPUs to use for training (use all by default)')
    parser_system.add_argument('--no-cuda',
                               default=False, action='store_true', help='run on CPU')
    parser_system.add_argument('--seed',
                               type=int, default=None, help='random seed')
    parser_system.add_argument('--log-interval',
                               default=10, type=int,
                               help='print loss/accuracy every N batches')
    parser_system.add_argument('-j', '--workers',
                               default=4, type=int,
                               help='workers used for train/val loaders')
    args = parser.parse_args()

    # Set up logging
    log_folder = set_up_logging(args)

    # Define Tensorboard SummaryWriter
    logging.info('Initialize Tensorboard SummaryWriter')
    writer = tensorboardX.SummaryWriter(log_dir=log_folder)

    # Set up env
    # Specify CUDA_VISIBLE_DEVICES and seeds
    set_up_env(args)

    if args.multi_run == None:

        try:
            main(args, writer, log_folder)
        except Exception as exp:
            if args.quiet:
                print('Unhandled error: {}'.format(repr(exp)))
            logging.error('Unhandled error: %s' % repr(exp))
            logging.error(traceback.format_exc())
            logging.error('Execution finished with errors :(')
            sys.exit(-1)
        finally:
            logging.shutdown()
            writer.close()
            print('All done! (logged to {}'.format(log_folder))

    else:
        train_scores = np.zeros((args.multi_run, args.epochs))
        val_scores = np.zeros((args.multi_run, args.epochs))
        test_scores = np.zeros((args.multi_run))

        for i in range(args.multi_run):
            logging.info('Multi-Run: {} of {}'.format(i + 1, args.multi_run))
            try:
                train_scores[i, :], val_scores[i, :], test_scores[i] = main(args, writer, log_folder, multi_run=i)

            except Exception as exp:
                if args.quiet:
                    print('Unhandled error: {}'.format(repr(exp)))
                logging.error('Unhandled error: %s' % repr(exp))
                logging.error(traceback.format_exc())
                logging.error('Execution finished with errors :(')
                sys.exit(-1)

        np.save(os.path.join(log_folder, 'train_values.npy'), train_scores)
        np.save(os.path.join(log_folder, 'val_values.npy'), val_scores)
        train_curve = plot_mean_variance(train_scores,
                                         suptitle='Multi-Run: Train',
                                         xlabel='Epochs', ylabel='Accuracy',
                                         ylim=[0, 100.0])
        writer.add_image('train_curve', train_curve)
        logging.info('Generated mean-variance plot for train')
        val_curve = plot_mean_variance(val_scores,
                                       suptitle='Multi-Run: Val',
                                       xlabel='Epochs', ylabel='Accuracy',
                                       ylim=[0, 100.0])
        writer.add_image('val_curve', train_curve)
        logging.info('Generated mean-variance plot for val')
        logging.info('Multi-run values for test-mean: {} test-std: {}'.format(np.mean(test_scores),
                                                                              np.std(test_scores)))
        logging.shutdown()
        writer.close()
        print('All done! (logged to {}'.format(log_folder))
