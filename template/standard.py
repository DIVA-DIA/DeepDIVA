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
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# DeepDIVA
import dataset
import models
from init.initializer import *
from template.standard.evaluate import test
from template.standard.evaluate import validate
from template.standard.train import train
from util.misc import checkpoint
from util.visualization.mean_std_plot import plot_mean_variance


#######################################################################################################################


def main(args, writer, log_folder, **kwargs):
    """This is the main routine where train(), validate() and test() are called."""

    # Get the selected model
    model_expected_input_size = models.__dict__[args.model]().expected_input_size
    logging.info('Model {} expects input size of {}'.format(args.model, model_expected_input_size))

    # Setting up the dataloaders
    train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size)

    # Setting up model, optimizer, criterion
    model, criterion, optimizer, best_value = set_up_model(num_classes)

    # Core routine
    logging.info('Begin training')
    val_value = np.zeros((args.epochs - args.start_epoch))
    train_value = np.zeros((args.epochs - args.start_epoch))

    validate(val_loader, model, criterion, writer, -1)
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_value[epoch] = train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)
        # Validate
        val_value[epoch] = validate(val_loader, model, criterion, writer, epoch, **kwargs)
        if args.decay_lr is not None:
            adjust_learning_rate(optimizer, epoch, args.decay_lr)
        best_value = checkpoint(epoch, val_value[epoch], best_value, model, optimizer, log_folder)

    # Test
    test_value = test(test_loader, model, criterion, writer, epoch, **kwargs)
    logging.info('Training completed')

    return train_value, val_value, test_value


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
    # If the dataset selected is a class in dataset, use it.
    if (args.dataset is not None) and (args.dataset in dataset.__dict__):
        logging.debug('Using an user definer class to load: ' + args.dataset)
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
    # Else, assume it is an image folder whose path is passed as 'args.dataset_folder'
    else:
        logging.debug('Using the image folder routine to load from: ' + args.dataset_folder)
        """
        Structure of the dataset expected
        
        Split folders 
        -------------
        'args.dataset_folder' has to point to the three folder train/val/test. 
        Example:  
        
        ~/../../data/svhn
        
        where the dataset_folder contains the splits sub-folders as follow:
        
        args.dataset_folder/train
        args.dataset_folder/val
        args.dataset_folder/test
        
        Classes folders
        ---------------
        In each of the three splits (train,val,test) should have different classes in a separate folder with the class 
        name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).
        Example:
        
        train/dog/whatever.png
        train/dog/you.png
        train/dog/like.png
        
        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932_.png
        """

        # Get the splits folders
        traindir = os.path.join(args.dataset_folder, 'train')
        valdir = os.path.join(args.dataset_folder, 'test')  # TODO change as soon as svhn has a val set :)
        testdir = os.path.join(args.dataset_folder, 'test')

        # Sanity check on the splits folders
        if not os.path.isdir(traindir):
            logging.error("Train folder not found in the args.dataset_folder=" + args.dataset_folder)
            sys.exit(-1)
        if not os.path.isdir(valdir):
            logging.error("Val folder not found in the args.dataset_folder=" + args.dataset_folder)
            sys.exit(-1)
        if not os.path.isdir(testdir):
            logging.error("Test folder not found in the args.dataset_folder=" + args.dataset_folder)
            sys.exit(-1)

        # Init the dataset splits
        train_ds = datasets.ImageFolder(traindir)
        val_ds = datasets.ImageFolder(valdir)
        test_ds = datasets.ImageFolder(testdir)

    # TODO what about the normalization?
    # train_ds.__getitem__(0) to get an image but its weird?
    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')
    train_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #       transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    val_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #      transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    test_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #     transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    # Setup dataloaders
    logging.debug('Setting up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             shuffle=False,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              shuffle=False,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # mean, std = _compute_mean_std(np.asarray([item[0] for item in train_ds.imgs]))

    return train_loader, val_loader, test_loader, len(train_ds.classes)


def _compute_mean_std(data):
    """
    Computes the mean and std for R,G,B channels
    :return:
    """
    mean = np.array([np.mean(data[:, :, :, 0]), np.mean(data[:, :, :, 1]),
                     np.mean(data[:, :, :, 2])]) / 255.0
    std = np.array([np.std(data[:, :, :, 0]), np.std(data[:, :, :, 1]),
                    np.std(data[:, :, :, 2])]) / 255.0
    return mean, std

#######################################################################################################################


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
                                  args.dataset if args.dataset in dataset.__dict__ else os.path.basename(
                                      os.path.normpath(args.dataset_folder)),
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


def adjust_learning_rate(optimizer, epoch, num_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    lr = args.lr * (0.1 ** (epoch // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    #TODO dataset and dataset-folder should never exist together
    parser_data.add_argument('--dataset',
                             choices=dataset_options,
                             help='which dataset to train/test on.')
    parser_data.add_argument('--dataset-folder',
                             help='location of the dataset on the machine e.g root/data',
                             default=None,
                             type=str)
    parser_data.add_argument('--log-dir',
                             help='where to save logs', default='./data/')
    parser_data.add_argument('--log-folder',
                             help='override default log folder (to resume logging of experiment). Normally you do not use this.',
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
