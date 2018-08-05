# Utils
import logging
import os
import sys

# Torch related stuff
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# DeepDIVA
import models
from datasets.image_folder_dataset import ImageFolderApply
from template.runner.triplet.transforms import MultiCrop
from template.setup import _load_mean_std_from_file, _get_optimizer, \
    _load_class_frequencies_weights_from_file


def set_up_dataloader(model_expected_input_size, dataset_folder, batch_size, workers, inmem,
                      multi_crop, classify, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    :param model_expected_input_size: tuple
        Specify the height and width that the model expects.

    :param dataset_folder: string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

    :param batch_size: int
        Number of datapoints to process at once

    :param workers: int
        Number of workers to use for the dataloaders

    :param inmem: boolean
        Flag: if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.

    :param multi_crop: int
        if None, the MultiCrop transform is not applied to the data. Otherwise, multi_crop contains
        an integer which specifies how many crops to make from each image.

    :param classify : boolean
            Specifies whether to generate a classification report for the data or not.

    :param kwargs: dict
        Any additional arguments.

    :return: dataloader, dataloader, dataloader, int
        Three dataloaders for train, val and test. Number of classes for the model.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    # Load the dataset as images
    apply_ds = ImageFolderApply(path=dataset_folder, classify=classify)

    # Loads the analytics csv and extract mean and std
    try:
        mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers)
    except:
        logging.error('analytics.csv not found in folder. Please copy the one generated in the '
                        'training folder to this folder.')
        logging.error('Currently normalizing with 0.5 for all channels for mean and std.')
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    if multi_crop == None:
        apply_ds.transform = transforms.Compose([
            transforms.Resize(model_expected_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        apply_ds.transform = transforms.Compose([
            MultiCrop(size=model_expected_input_size, n_crops=multi_crop),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
        ])
    apply_loader = torch.utils.data.DataLoader(apply_ds,
                                               shuffle=False,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)

    return apply_loader, len(apply_ds.classes)


#
# def set_up_model(output_channels, model_name, pretrained, optimizer_name, no_cuda, resume, load_model, start_epoch,
#                  train_loader, disable_databalancing, dataset_folder, inmem, workers, num_classes=None, **kwargs):
#     """
#     Instantiate model, optimizer, criterion. Load a pretrained model or resume from a checkpoint.
#
#     Parameters
#     ----------
#     output_channels : int
#         Specify shape of final layer of network. Only used if num_classes is not specified.
#
#     model_name : string
#         Name of the model
#
#     pretrained : bool
#         Specify whether to load a pretrained model or not
#
#     optimizer_name : string
#         Name of the optimizer
#
#     lr: float
#         Value for learning rate
#
#     no_cuda : bool
#         Specify whether to use the GPU or not
#
#     resume : string
#         Path to a saved checkpoint
#
#     load_model : string
#         Path to a saved model
#
#     start_epoch : int
#         Epoch from which to resume training. If if not resuming a previous experiment the value is 0
#
#     num_classes: int
#         Number of classes for the model
#
#     kwargs: dict
#         Any additional arguments.
#
#     Returns
#     -------
#         model, criterion, optimizer, best_value, start_epoch
#     """
#
#     # Initialize the model
#     logging.info('Setting up model {}'.format(model_name))
#
#     output_channels = output_channels if num_classes == None else num_classes
#     model = models.__dict__[model_name](output_channels=output_channels, pretrained=pretrained)
#
#     # Get the optimizer created with the specified parameters in kwargs (such as lr, momentum, ... )
#     optimizer = _get_optimizer(optimizer_name, model, **kwargs)
#
#     # Get the criterion
#     if disable_databalancing:
#         criterion = nn.CrossEntropyLoss()
#     else:
#         try:
#             weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers)
#             criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).type(torch.FloatTensor))
#             logging.info('Loading weights for data balancing')
#         except:
#             logging.warning('Unable to load information for data balancing. Using normal criterion')
#             criterion = nn.CrossEntropyLoss()
#
#     # Transfer model to GPU (if desired)
#     if not no_cuda:
#         logging.info('Transfer model to GPU')
#         model = torch.nn.DataParallel(model).cuda()
#         criterion = criterion.cuda()
#         cudnn.benchmark = True
#
#     # Load saved model
#     if load_model:
#         if os.path.isfile(load_model):
#             model_dict = torch.load(load_model)
#             logging.info('Loading a saved model')
#             try:
#                 model.load_state_dict(model_dict['state_dict'], strict=False)
#             except Exception as exp:
#                 logging.warning(exp)
#         else:
#             logging.error("No model dict found at '{}'".format(load_model))
#             sys.exit(-1)
#
#     # Resume from checkpoint
#     if resume:
#         if os.path.isfile(resume):
#             logging.info("Loading checkpoint '{}'".format(resume))
#             checkpoint = torch.load(resume)
#             start_epoch = checkpoint['epoch']
#             best_value = checkpoint['best_value']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             # val_losses = [checkpoint['val_loss']] #not used?
#             logging.info("Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
#         else:
#             logging.error("No checkpoint found at '{}'".format(resume))
#             sys.exit(-1)
#     else:
#         best_value = 0.0
#
#     return model, criterion, optimizer, best_value, start_epoch
