# Utils
import argparse
import os

# Torch
import torch

# DeepDIVA
import models
import sys

def parse_arguments(args=None):
    """
    Argument Parser
    """

    ###############################################################################
    # Parsers
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Template for training a network on a dataset')

    # Add all options
    _general_parameters(parser)
    _data_options(parser)
    _training_options(parser)
    _apply_options(parser)
    _optimizer_options(parser)
    _system_options(parser)
    _triplet_options(parser)
    _semantic_segmentation_options(parser)

    ###############################################################################
    # Parse argument
    args = parser.parse_args(args)

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(args.dataset_folder))

    # If contains 'bd' override the runner class to bidimensional
    if 'bd_' in dataset and args.runner_class == 'image_classification':
        args.runner_class = 'bidimensional'

    # If experiment name is not set, ask for one
    if args.experiment_name is None:
        args.experiment_name = input("Please enter an experiment name:")

    return args, parser


def _general_parameters(parser):
    """
    General options
    """
    # List of possible custom runner class. A runner class is defined as a module in template.runner
    runner_class_options = ["image_classification", "point_cloud", "triplet",
                            "apply_model", "image_auto_encoding",
                            "semantic_segmentation"]

    parser_general = parser.add_argument_group('GENERAL', 'General Options')
    parser_general.add_argument('--experiment-name',
                                type=str,
                                default=None,
                                help='provide a meaningful and descriptive name to this run')
    parser_general.add_argument('--quiet',
                                action='store_true',
                                help='Do not print to stdout (log only).')
    parser_general.add_argument('--debug',
                                default=False,
                                action='store_true',
                                help='log debug level messages')
    parser_general.add_argument('--multi-run',
                                type=int,
                                default=None,
                                help='run main N times with different random seeds')
    parser_general.add_argument('--hyper-param-optim',
                                type=str,
                                default=None,
                                help='path to a JSON file containing all variable names (as defined in the argument '
                                     'parser) that need to be searched over.')
    parser_general.add_argument('--sig-opt',
                                type=str,
                                default=None,
                                help='path to a JSON file containing sig_opt variables and sig_opt bounds.')
    parser_general.add_argument('--sig-opt-token',
                                type=str,
                                default=None,
                                help='place your SigOpt API token here.')
    parser_general.add_argument('--sig-opt-runs',
                                type=int,
                                default=100,
                                help='number of updates of SigOpt required')
    parser_general.add_argument('--runner-class',
                                choices=runner_class_options,
                                default="image_classification",
                                help='which runner class to use.')
    parser_general.add_argument('--ignoregit',
                                action='store_true',
                                help='Run irrespective of git status.')


def _data_options(parser):
    """
    Defines all parameters relative to the data.
    """
    # List of possible custom dataset already implemented
    parser_data = parser.add_argument_group('DATA', 'Dataset Options')
    parser_data.add_argument('--dataset-folder',
                             help='location of the dataset on the machine e.g root/data',
                             required=True)
    parser_data.add_argument('--inmem',
                             default=False,
                             action='store_true',
                             help='Attempt to load the entire image dataset in memory')
    parser_data.add_argument('--disable-databalancing',
                             default=False,
                             action='store_true',
                             help='Suppress data balancing')
    parser_data.add_argument('--output-folder',
                             default='./output/',
                             help='where to save all output files.', )
    parser_data.add_argument('--disable-dataset-integrity',
                             default=False,
                             action='store_true',
                             help='Suppress the dataset integrity verification')
    parser_data.add_argument('--enable-deep-dataset-integrity',
                             default=False,
                             action='store_true',
                             help='Enable the deep dataset integrity verification')


def _training_options(parser):
    """
    Training options
    """
    # List of possible custom models already implemented
    # NOTE: If a model is missing and you get a argument parser error: check in the init file of models if its there!
    model_options = [name for name in models.__dict__ if callable(models.__dict__[name])]

    parser_train = parser.add_argument_group('TRAIN', 'Training Options')
    parser_train.add_argument('--model-name',
                              type=str,
                              choices=model_options,
                              default='CNN_basic',
                              help='which model to use for training')
    parser_train.add_argument('--batch-size',
                              type=int,
                              default=64,
                              help='input batch size for training')
    parser_train.add_argument('--epochs',
                              type=int,
                              default=5,
                              help='how many epochs to train')
    parser_train.add_argument('--pretrained',
                              action='store_true',
                              default=False,
                              help='use pretrained model. (Not applicable for all models)')
    parser_train.add_argument('--load-model',
                              type=str,
                              default=None,
                              help='path to latest checkpoint')
    parser_train.add_argument('--resume',
                              type=str,
                              default=None,
                              help='path to latest checkpoint')
    parser_train.add_argument('--start-epoch',
                              type=int,
                              metavar='N',
                              default=0,
                              help='manual epoch number (useful on restarts)')
    parser_train.add_argument('--validation-interval',
                              type=int,
                              default=1,
                              help='run evaluation on validation set every N epochs')
    parser_train.add_argument('--checkpoint-all-epochs',
                              action='store_true',
                              default=False,
                              help='make a checkpoint after every epoch')

def _apply_options(parser):
    """
    Options specific for applying a model
    """
    parser_apply = parser.add_argument_group('APPLY', 'Apply Model Options')

    parser_apply.add_argument('--classify',
                              action='store_true',
                              default=False,
                              help='run on generate classification report on the dataset')
    parser_apply.add_argument('--multi-crop',
                              type=int,
                              default=None,
                              help='generate multiple crops out of each input image, apply model and average over all crops of image')
    parser_apply.add_argument('--output-channels',
                              type=int,
                              default=None,
                              help='override the number of output channels for loading specific models')


def _optimizer_options(parser):
    """
    Options specific for optimizers
    """
    # List of possible optimizers already implemented in PyTorch
    optimizer_options = [name for name in torch.optim.__dict__ if callable(torch.optim.__dict__[name])]

    parser_optimizer = parser.add_argument_group('OPTIMIZER', 'Optimizer Options')

    parser_optimizer.add_argument('--optimizer-name',
                                  choices=optimizer_options,
                                  default='SGD',
                                  help='optimizer to be used for training')
    parser_optimizer.add_argument('--lr',
                                  type=float,
                                  default=0.001,
                                  help='learning rate to be used for training')
    parser_optimizer.add_argument('--decay-lr',
                                  type=int,
                                  default=None,
                                  help='drop LR by 10 every N epochs')
    parser_optimizer.add_argument('--momentum',
                                  type=float,
                                  default=0,
                                  help='momentum (parameter for the optimizer)')
    parser_optimizer.add_argument('--dampening',
                                  type=float,
                                  default=0,
                                  help='dampening (parameter for the SGD)')
    parser_optimizer.add_argument('--weight-decay',
                                  type=float,
                                  default=0,
                                  help='weight_decay coefficient, also known as L2 regularization')


def _system_options(parser):
    """
    System options
    """
    parser_system = parser.add_argument_group('SYS', 'System Options')
    parser_system.add_argument('--gpu-id',
                               default=None,
                               help='which GPUs to use for training (use all by default)')
    parser_system.add_argument('--no-cuda',
                               action='store_true',
                               default=False,
                               help='run on CPU')
    parser_system.add_argument('--seed',
                               type=int,
                               default=None,
                               help='random seed')
    parser_system.add_argument('--log-interval',
                               type=int,
                               default=20,
                               help='print loss/accuracy every N batches')
    parser_system.add_argument('-j', '--workers',
                               type=int,
                               default=4,
                               help='workers used for train/val loaders')


def _triplet_options(parser):
    """
    Triplet options

    These parameters are used by the runner class template.runner.triplet
    """
    parser_triplet = parser.add_argument_group('TRIPLET', 'Triplet Options')
    parser_triplet.add_argument('--n-triplets',
                                type=int,
                                default=1280000, metavar='N',
                                help='how many triplets to generate from the dataset')
    parser_triplet.add_argument('--margin',
                                type=float,
                                default=2.0,
                                help='the margin value for the triplet loss function')
    parser_triplet.add_argument('--anchor-swap',
                                action='store_true',
                                help='turns on anchor swap')
    parser_triplet.add_argument('--map',
                                type=str,
                                default='full',
                                help='switch between "auto", "full" or specify K for AP@K')
    parser_triplet.add_argument('--regenerate-every',
                                type=int,
                                default=5, metavar='N',
                                help='re-generate triplets every N epochs')


def _semantic_segmentation_options(parser):
    """
    Triplet options

    These parameters are used by the runner class template.runner.semantic_segmentation
    """
    semantic_segmentation = parser.add_argument_group('Semantic', 'Semantic Segmentation')
    semantic_segmentation.add_argument('--input-patch-size',
                                type=int,
                                default=256, metavar='N',
                                help='size of the square input patch e.g. with 32 the input will be re-sized to 32x32')

