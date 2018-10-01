"""
General purpose utility functions.

"""

# Utils
import logging
import os
import os.path
import shutil
import string

import cv2
import numpy as np
import torch


def _prettyprint_logging_label(logging_label):
    """Format the logging label in a pretty manner.

    Parameters
    ----------
    logging_label : str
        The label used in logging

    Returns
    -------
    logging_label : str
        Correctly formatted logging label.

    """
    if len(logging_label) < 5:
        for i in range(5 - len(logging_label)):
            logging_label = logging_label + ' '
    return logging_label


class AverageMeter(object):
    """Computes and stores the average and current value

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    """
    Computes the accuracy@K for the specified values of K

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    :param output:
        The output of the model

    :param target:
        The GT for the corresponding output

    :param topk:
        Top@k return value. It can be a tuple (1,5) and it return Top1 and Top5

    :return:
        Top@k accuracy
    """


def adjust_learning_rate(lr, optimizer, epoch, decay_lr_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs.

    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    lr : float
        Learning rate.
    optimizer : torch.optim object
        The optimizer used for training the network.
    epoch : int
        Current training epoch.
    decay_lr_epochs : int
        Change the learning rate every N epochs.

    Returns
    -------
    None

    """
    import copy
    original_lr = copy.deepcopy(lr)
    lr = lr * (0.1 ** (epoch // decay_lr_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if original_lr != lr:
        logging.info('Learning rate decayed. New learning rate is: {}'.format(lr))
    return


def checkpoint(epoch, new_value, best_value, model, optimizer, log_dir,
               invert_best=False, checkpoint_all_epochs=False):
    """Saves the current training checkpoint and the best valued checkpoint to file.

    Parameters
    ----------
    epoch : int
        Current epoch, for logging purpose only.
    new_value : float
        Current value achieved by the model at this epoch.
        To be compared with 'best_value'.
    best_value : float
        Best value ever obtained (so the last checkpointed model).
        To be compared with 'new_value'.
    model : torch.nn.module object
        The model we are checkpointing, this can be saved on file if necessary.
    optimizer :
        The optimizer that is being used to train this model.
        It is necessary if we were to resume the training from a checkpoint.
    log_dir : str
        Output folder where to put the model.
    invert_best : bool
        Changes the scale such that smaller values are better than bigger values
        (useful when metric evaluted is error rate)
    checkpoint_all_epochs : bool
        If enabled, save checkpoint after every epoch.

    Returns
    -------
    best_value : float
        Best value ever obtained.

    """
    if invert_best:
        is_best = new_value < best_value
        best_value = min(new_value, best_value)
    else:
        is_best = new_value > best_value
        best_value = max(new_value, best_value)
    filename = os.path.join(log_dir, 'checkpoint.pth.tar')
    torch.save({
        'epoch': epoch + 1,
        'arch': str(type(model)),
        'state_dict': model.state_dict(),
        'best_value': best_value,
        'optimizer': optimizer.state_dict(),
    }, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
    # If enabled, save all checkpoints with epoch number.
    if checkpoint_all_epochs == True:
        shutil.move(filename, os.path.join(os.path.split(filename)[0], 'checkpoint_{}.pth.tar'.format(epoch)))
    return best_value


def to_capital_camel_case(s):
    """Converts a string to camel case.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Input string `s` converted to camel case.

    """
    return s[0].capitalize() + string.capwords(s, sep='_').replace('_', '')[1:] if s else s


def get_all_files_in_folders_and_subfolders(root_dir=None):
    """Get all the files in a folder and sub-folders.

    Parameters
    ----------
    root_dir : str
        All files in this directory and it's sub-folders will be returned by this method.

    Returns
    -------
    paths : list of str
        List of paths to all files in this folder and it's subfolders.
    """
    paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def save_image_and_log_to_tensorboard(writer=None, tag=None, image_tensor=None, global_step=None):
    """Utility function to save image in the output folder and also log it to Tensorboard.

    Parameters
    ----------
    writer : tensorboardX.writer.SummaryWriter object
        The writer object for Tensorboard
    tag : str
        Name of the image.
    image_tensor : ndarray [H x W x C]
        Image to be saved and logged to Tensorboard.
    global_step : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """

    # Log image to Tensorboard
    writer.add_image(tag=tag, img_tensor=image_tensor, global_step=global_step)

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag + '_{}.png'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag + '.png')

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    # Write image to output folder
    cv2.imwrite(dest_filename, np.roll(image_tensor, 1, axis=-1))

    return

def has_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.

    Parameters
    ----------
    filename : string
        path to a file
    extensions : list
        extensions to match against
    Returns
    -------
    bool
        True if the filename ends with one of given extensions, false otherwise.
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


