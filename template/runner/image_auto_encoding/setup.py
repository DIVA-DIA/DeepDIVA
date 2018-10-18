# Utils
import logging
import os
import sys

# Torch related stuff
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

# DeepDIVA
import models
from template.setup import _get_optimizer, verify_integrity_deep, verify_integrity_quick, _dataloaders_from_datasets, \
    image_folder_dataset, _load_mean_std_from_file, transforms


def set_up_model(output_channels, model_name, pretrained, optimizer_name, no_cuda, resume, load_model,
                 start_epoch, disable_databalancing, dataset_folder, inmem, workers, num_classes=None,
                 **kwargs):
    """
    Instantiate model, optimizer, criterion. Load a pretrained model or resume from a checkpoint.

    Parameters
    ----------
    output_channels : int
        Specify shape of final layer of network. Only used if num_classes is not specified.
    model_name : string
        Name of the model
    pretrained : bool
        Specify whether to load a pretrained model or not
    optimizer_name : string
        Name of the optimizer
    no_cuda : bool
        Specify whether to use the GPU or not
    resume : string
        Path to a saved checkpoint
    load_model : string
        Path to a saved model
    start_epoch : int
        Epoch from which to resume training. If if not resuming a previous experiment the value is 0
    disable_databalancing : boolean
        If True the criterion will not be fed with the class frequencies. Use with care.
    dataset_folder : String
        Location of the dataset on the file system
    inmem : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.
    workers : int
        Number of workers to use for the dataloaders
    num_classes: int
        Number of classes for the model

    Returns
    -------
    model : nn.Module
        The actual model
    criterion : nn.loss
        The criterion for the network
    optimizer : torch.optim
        The optimizer for the model
    best_value : float
        Specifies the former best value obtained by the model.
        Relevant only if you are resuming training.
    start_epoch : int
        Specifies at which epoch was the model saved.
        Relevant only if you are resuming training.
    """

    # Initialize the model
    logging.info('Setting up model {}'.format(model_name))

    output_channels = output_channels if num_classes == None else num_classes
    model = models.__dict__[model_name](output_channels=output_channels,
                                        pretrained=pretrained,
                                        auto_encoder_mode=True,
                                        **kwargs)

    # Get the optimizer created with the specified parameters in kwargs (such as lr, momentum, ... )
    optimizer = _get_optimizer(optimizer_name, model, **kwargs)

    # Get the criterion
    # TODO: parameterize this out
    criterion = nn.MSELoss()

    # Transfer model to GPU (if desired)
    if not no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Load saved model
    if load_model:
        if os.path.isfile(load_model):
            model_dict = torch.load(load_model)
            logging.info('Loading a saved model')
            try:
                model.load_state_dict(model_dict['state_dict'], strict=False)
            except Exception as exp:
                logging.warning(exp)
        else:
            logging.error("No model dict found at '{}'".format(load_model))
            sys.exit(-1)

    # Resume from checkpoint
    if resume:
        if os.path.isfile(resume):
            logging.info("Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_value = checkpoint['best_value']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # val_losses = [checkpoint['val_loss']] #not used?
            logging.info("Loaded checkpoint '{}' (epoch {})"
                         .format(resume, checkpoint['epoch']))
        else:
            logging.error("No checkpoint found at '{}'".format(resume))
            sys.exit(-1)
    else:
        best_value = 0.0

    return model, criterion, optimizer, best_value, start_epoch

def set_up_dataloaders(model_expected_input_size, dataset_folder, batch_size, workers,
                       disable_dataset_integrity, enable_deep_dataset_integrity,  inmem=False, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    model_expected_input_size : tuple
        Specify the height and width that the model expects.
    dataset_folder : string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn
    batch_size : int
        Number of datapoints to process at once
    workers : int
        Number of workers to use for the dataloaders
    inmem : boolean
        Flag: if False, the dataset is loaded in an online fashion i.e. only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        Dataloaders for train, val and test.
    int
        Number of classes for the model.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    ###############################################################################################
    # Verify dataset integrity
    if not disable_dataset_integrity:
        if enable_deep_dataset_integrity:
            if not verify_integrity_deep(dataset_folder):
                sys.exit(-1)
        else:
            if not verify_integrity_quick(dataset_folder):
                sys.exit(-1)

    ###############################################################################################
    # Load the dataset splits as images
    try:
        logging.debug("Try to load dataset as images")
        train_ds, val_ds, test_ds = image_folder_dataset.load_dataset(dataset_folder, inmem, workers)

        # Loads the analytics csv and extract mean and std
        mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers)

        # Set up dataset transforms
        logging.debug('Setting up dataset transforms')
        transform = transforms.Compose([
            transforms.Resize(model_expected_input_size),
            transforms.ToTensor(),
        ])

        train_ds.transform = transform
        val_ds.transform = transform
        test_ds.transform = transform

        train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size, train_ds, val_ds, test_ds,
                                                                           workers)
        logging.info("Dataset loaded as images")
        return train_loader, val_loader, test_loader, len(train_ds.classes)

    except RuntimeError:
        logging.debug("No images found in dataset folder provided")

    ###############################################################################################
    # Verify that eventually a dataset has been correctly loaded
    logging.error("No datasets have been loaded. Verify dataset folder location or dataset folder structure")
    sys.exit(-1)