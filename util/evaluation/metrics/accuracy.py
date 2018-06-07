# Utils
import logging


def accuracy(predicted, target, topk=(1,)):
    """Computes the accuracy@K for the specified values of K

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    predicted : torch.FloatTensor
        The predicted output values of the model.
        The size is batch_size x num_classes
    target : torch.LongTensor
        The ground truth for the corresponding output.
        The size is batch_size x 1
    topk : tuple
        Multiple values for K can be specified in a tuple, and the
        different accuracies@K will be computed.

    Returns
    -------
    res : list
        List of accuracies computed at the different K's specified in `topk`

    """
    if len(predicted.shape) != 2:
        logging.error('Invalid input shape for prediction: predicted.shape={}'
                      .format(predicted.shape))
        return None
    if len(target.shape) != 1:
        logging.error('Invalid input shape for target: target.shape={}'
                      .format(target.shape))
        return None

    if len(predicted) == 0 or len(target) == 0 or len(predicted) != len(target):
        logging.error('Invalid input for accuracy: len(predicted)={}, len(target)={}'
                      .format(len(predicted), len(target)))
        return None

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = predicted.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
