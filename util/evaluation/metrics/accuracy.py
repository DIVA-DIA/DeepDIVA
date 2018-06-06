def accuracy(predicted, target, topk=(1,)):
    """Computes the accuracy@K for the specified values of K

    From https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    predicted : torch.FloatTensor
        The predicted output values of the model.
    target : torch.LongTensor
        The ground truth for the corresponding output.
    topk : tuple
        Multiple values for K can be specified in a tuple, and the
        different accuracies@K will be computed.

    Returns
    -------
    res : list
        List of accuracies computed at the different K's specified in `topk`

    """

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
