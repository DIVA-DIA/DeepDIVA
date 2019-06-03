# Utils
import logging
import os
import sys

# Torch related stuff
import torch.utils.data
import torchvision.transforms as transforms

# DeepDIVA
from datasets import multi_label_image_folder_dataset
from template.setup import _load_mean_std_from_file, _dataloaders_from_datasets, _verify_dataset_integrity


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
    # Load the dataset splits as images
    try:
        logging.debug("Try to load dataset as multi-label-images")
        train_ds, val_ds, test_ds = multi_label_image_folder_dataset.load_dataset(dataset_folder, inmem, workers)

        # Loads the analytics csv and extract mean and std
        mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers, kwargs['runner_class'])

        # Set up dataset transforms
        logging.debug('Setting up dataset transforms')
        transform = transforms.Compose([
            transforms.Resize(model_expected_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_ds.transform = transform
        val_ds.transform = transform
        test_ds.transform = transform

        train_loader, val_loader, test_loader = _dataloaders_from_datasets(batch_size, train_ds, val_ds, test_ds,
                                                                           workers)
        logging.info("Dataset loaded as images")
        _verify_dataset_integrity(dataset_folder, disable_dataset_integrity, enable_deep_dataset_integrity)
        return train_loader, val_loader, test_loader, len(train_ds.classes)

    except RuntimeError:
        logging.debug("No images found in dataset folder provided")
    # Verify that eventually a dataset has been correctly loaded
    logging.error("No datasets have been loaded. Verify dataset folder location or dataset folder structure")
    sys.exit(-1)
