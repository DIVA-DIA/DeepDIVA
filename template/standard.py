"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

# Utils
import argparse
import os
import sys
import traceback

# Tensor board
import tensorboardX

# DeepDIVA
import datasets
import models
from init.initializer import *
from template.standard.evaluate import validate, test
from template.standard.setup import set_up_env, set_up_logging, set_up_model, set_up_dataloaders
from template.standard.train import train
from util.misc import checkpoint, adjust_learning_rate
from util.visualization.mean_std_plot import plot_mean_variance


#######################################################################################################################


def main(writer, log_folder, model_name, epochs, decay_lr, lr, **kwargs):
    """
    This is the main routine where train(), validate() and test() are called.
    :param writer: Tensorboard SummaryWriter
        Responsible for writing logs in Tensorboard compatible format.
    :param log_folder: string
        Path to where logs/checkpoints are saved
    :param model_name: string
        Name of the model
    :param epochs: int
        Number of epochs to train
    :param decay_lr: int N
        Decay the learning rate by a factor of 10 every N epochs
    :param lr: float
        Value for learning rate
    :param kwargs: dict
        Any additional arguments.
    :return: train_precs, val_precs, test_prec
        Precision values for train and validation splits. Single precision value for the test split.
    """

    # Get the selected model
    model_expected_input_size = models.__dict__[model_name]().expected_input_size
    logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

    # Setting up the dataloaders
    train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size, **kwargs)

    # Setting up model, optimizer, criterion
    model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                       model_name=model_name,
                                                                       lr=lr, **kwargs)


    # Core routine
    logging.info('Begin training')
    val_precs = np.zeros((epochs - start_epoch))
    train_precs = np.zeros((epochs - start_epoch))

    validate(val_loader, model, criterion, writer, -1)
    for epoch in range(start_epoch, epochs):
        # Train
        train_precs[epoch] = train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)
        # Validate
        val_precs[epoch] = validate(val_loader, model, criterion, writer, epoch, **kwargs)
        if args.decay_lr is not None:
            adjust_learning_rate(lr, optimizer, epoch, decay_lr)
        best_value = checkpoint(epoch, val_precs[epoch], best_value, model, optimizer, log_folder)

    # Test
    test_prec = test(test_loader, model, criterion, writer, epoch, **kwargs)
    logging.info('Training completed')

    return train_precs, val_precs, test_prec


#######################################################################################################################

if __name__ == "__main__":

    model_options = [name for name in models.__dict__ if callable(models.__dict__[name])]
    dataset_options = [name for name in datasets.__dict__ if callable(datasets.__dict__[name])]
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
                              dest='model_name',
                              choices=model_options,
                              help='which model to use for training',
                              type=str, default='CNN_basic')
    parser_train.add_argument('--lr',
                              help='learning rate to be used for training',
                              type=float, default=0.001)
    parser_train.add_argument('--optimizer',
                              dest='optimizer_name',
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
    args.__dict__['log_folder'] = set_up_logging(args_dict=args.__dict__, **args.__dict__)

    # Define Tensorboard SummaryWriter
    logging.info('Initialize Tensorboard SummaryWriter')
    writer = tensorboardX.SummaryWriter(log_dir=args.log_folder)

    # Set up execution environment
    # Specify CUDA_VISIBLE_DEVICES and seeds
    set_up_env(**args.__dict__)

    if args.multi_run == None:

        try:
            main(writer, **args.__dict__)
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
            print('All done! (logged to {}'.format(args.log_folder))

    else:
        train_scores = np.zeros((args.multi_run, args.epochs))
        val_scores = np.zeros((args.multi_run, args.epochs))
        test_scores = np.zeros((args.multi_run))

        for i in range(args.multi_run):
            logging.info('Multi-Run: {} of {}'.format(i + 1, args.multi_run))
            try:
                train_scores[i, :], val_scores[i, :], test_scores[i] = main(writer, run=i,
                                                                            **args.__dict__)
                train_curve = plot_mean_variance(train_scores[:i],
                                                 suptitle='Multi-Run: Train',
                                                 title='Runs: {}'.format(i+1),
                                                 xlabel='Epochs', ylabel='Accuracy',
                                                 ylim=[0, 100.0])
                writer.add_image('train_curve', train_curve)
                logging.info('Generated mean-variance plot for train')
                val_curve = plot_mean_variance(val_scores[:i],
                                               suptitle='Multi-Run: Val',
                                               title='Runs: {}'.format(i+1),
                                               xlabel='Epochs', ylabel='Accuracy',
                                               ylim=[0, 100.0])
                writer.add_image('val_curve', val_curve)
                logging.info('Generated mean-variance plot for val')

            except Exception as exp:
                if args.quiet:
                    print('Unhandled error: {}'.format(repr(exp)))
                logging.error('Unhandled error: %s' % repr(exp))
                logging.error(traceback.format_exc())
                logging.error('Execution finished with errors :(')
                sys.exit(-1)

        np.save(os.path.join(args.log_folder, 'train_values.npy'), train_scores)
        np.save(os.path.join(args.log_folder, 'val_values.npy'), val_scores)
        logging.info('Multi-run values for test-mean: {} test-std: {}'.format(np.mean(test_scores),
                                                                              np.std(test_scores)))
        logging.shutdown()
        writer.close()
        print('All done! (logged to {}'.format(args.log_folder))
