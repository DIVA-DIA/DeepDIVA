"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath, Michele Alberti
"""

# Utils
import argparse
import json
import os
import random
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
import models
from init.initializer import *
from template.standard.evaluate import test
from template.standard.evaluate import validate
from template.standard.train import train
from util.misc import checkpoint


#######################################################################################################################


def main(args, writer, log_folder):
    """This is the main routine where train(), validate() and test() are called."""

    # Get the selected model
    model_expected_input_size = models.__dict__[args.model]().expected_input_size
    logging.info('Model {} expects input size of {}'.format(args.model, model_expected_input_size))

    # Setting up the dataloaders
    train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size)

    # Setting up model, optimizer, criterion
    model, criterion, optimizer, best_prec = set_up_model(num_classes)

    # Train
    logging.info('Begin training')
    for epoch in range(args.start_epoch, args.epochs):
        # Validate
        val_prec = validate(val_loader, model, criterion, writer, epoch)
        train(train_loader, model, criterion, optimizer, writer, epoch)
        if args.decay_lr is not None:
            adjust_learning_rate(optimizer, epoch, args.decay_lr)
        checkpoint(epoch, val_prec, best_prec, model, optimizer, log_folder)

    # Test
    test(test_loader, model, criterion, writer, epoch)
    logging.info('Training completed')

    return


#######################################################################################################################


def set_up_model(num_classes):
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
            best_value = checkpoint['best_value']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            val_losses = [checkpoint['val_loss']]
            logging.info("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logging.error("No checkpoint found at '{}'".format(args.resume))
            sys.exit(-1)
    else:
        best_value = 0.0
    return model, criterion, optimizer, best_value


def set_up_dataloaders(model_expected_input_size):
    """Set up datasets and dataloaders"""

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


def adjust_learning_rate(optimizer, epoch, num_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    lr = args.lr * (0.1 ** (epoch // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#######################################################################################################################


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
                              type=str, default='CNN_basic')
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

    # Experiment name override
    if args.experiment_name is None:
        vars(args)['experiment_name'] = input("Experiment name:")

    ####################################################################################################################
    # Seed the random

    if args.seed:
        # Python
        random.seed(args.seed)

        # Numpy random
        np.random.seed(args.seed)

        # Torch random
        torch.manual_seed(args.seed)
        if not args.no_cuda:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.enabled = False

    ####################################################################################################################
    # Setup Logging
    basename = args.log_dir
    experiment_name = args.experiment_name
    if not args.log_folder:
        log_folder = os.path.join(basename, experiment_name, '{}'.format(time.strftime('%y-%m-%d-%Hh-%Mm-%Ss')))
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

    # Define Tensorboard SummaryWriter
    logging.info('Initialize Tensorboard SummaryWriter')
    writer = tensorboardX.SummaryWriter(log_dir=log_folder)



    # Set up logging to console
    if not args.quiet:
        fmtr = logging.Formatter(fmt='%(funcName)s %(levelname)s: %(message)s')
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = fmtr
        logging.getLogger().addHandler(stderr_handler)
        logging.info('Printing activity to the console')

    ####################################################################################################################

    # Set visible GPUs
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

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
