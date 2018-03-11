"""
General purpose utility code from https://raw.githubusercontent.com/pytorch/vision/master/torchvision/datasets/utils.py
"""

# Utils
import errno
import hashlib
import logging
import os
import os.path
import shutil
# Torch
import string

import torch


# TODO comment this
def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


#TODO comment this
def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy@k for the specified values of k

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

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, optimizer, epoch, num_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    lr = lr * (0.1 ** (epoch // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('Learning rate decayed. New learning rate is: {}'.format(lr))


def checkpoint(epoch, new_value, best_value, model, optimizer, log_dir):
    """
    Checks whether the checkpoint. If the model is better, it saves it to file.

    Parameters
    ----------
    :param epoch : int
        Current epoch, for logging purpose only

    :param new_value : float
        Current value achieved by the model at this epoch. To be compared with 'best_value'.

    :param best_value : float
        Best value every obtained (so the last checkpointed model). To be compared with 'new_value'.

    :param model:
        The model we are checkpointing. To be saved on file if necessary.

    :param optimizer:
        The optimizer we used to obtain this model. It is necessary if we were to resume the training from a checkpoint.

    :param log_dir:
        Output folder where to put the model.

    :return:
        None
    """
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

    return best_value


def to_capital_camel_case(s):
    return s[0].capitalize() + string.capwords(s, sep='_').replace('_', '')[1:] if s else s


def adjust_learning_rate(optimizer, lr, lr_decay=1e-6):
    """
    Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    Original implementation from: https://github.com/vbalnt/tfeat
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = lr / (1 + group['step'] * lr_decay)
