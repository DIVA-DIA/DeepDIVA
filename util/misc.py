"""
General purpose utility code from https://raw.githubusercontent.com/pytorch/vision/master/torchvision/datasets/utils.py
"""

# Utils
import errno
import hashlib
import os
import os.path
import shutil

# Torch
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


def checkpoint(epoch, new_value, best_value, model, optimizer, log_folder):
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

    :param log_folder:
        Output folder where to put the model.

    :return:
        None
    """
    is_best = new_value > best_value
    best_value = max(new_value, best_value)
    filename = os.path.join(log_folder, 'checkpoint.pth.tar')
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
