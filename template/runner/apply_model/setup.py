# Utils
import logging
import os

# Torch related stuff
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# DeepDIVA
from datasets.image_folder_dataset import ImageFolderApply
from template.runner.triplet.transforms import MultiCrop
from template.setup import _load_mean_std_from_file


def set_up_dataloader(model_expected_input_size, dataset_folder, batch_size, workers, inmem, multi_crop, **kwargs):
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

    :param kwargs: dict
        Any additional arguments.

    :return: dataloader, dataloader, dataloader, int
        Three dataloaders for train, val and test. Number of classes for the model.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    # Load the dataset as images
    apply_ds = ImageFolderApply(path=dataset_folder)

    # Loads the analytics csv and extract mean and std
    mean, std = _load_mean_std_from_file(dataset_folder, inmem, workers)

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
