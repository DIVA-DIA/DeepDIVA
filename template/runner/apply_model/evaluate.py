# Utils
import logging

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Torch related stuff
from torch.autograd import Variable

# DeepDIVA
from util.visualization.confusion_matrix_heatmap import make_heatmap


def feature_extract(data_loader, model, writer, epoch, no_cuda, log_interval, classify, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    :param data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set

    :param model : torch.nn.module
        The network model being used

    :param writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    :param epoch : int
        Number of the epoch (for logging purposes)

    :param no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    :param classify : boolean
        Specifies whether to generate a classification report for the data or not.

    :param log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    :return:
        None
    """
    logging_label = 'apply'

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    labels, features, preds, filenames = [], [], [], []

    multi_crop = False
    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=200)
    for batch_idx, (data, label, filename) in pbar:
        if len(data.size()) == 5:
            multi_crop = True
            bs, ncrops, c, h, w = data.size()
            data = data.view(-1, c, h, w)
        if not no_cuda:
            data = data.cuda()

        data_a, label = Variable(data, volatile=True), Variable(label)

        # Compute output
        out = model(data_a)

        if multi_crop:
            out = out.view(bs, ncrops, -1).mean(1)

        preds.append([np.argmax(item.data.cpu().numpy()) for item in out])
        features.append(out.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        filenames.append(filename)

        # Log progress to console
        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label + ' Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))

    # Measure accuracy (FPR95)
    num_tests = len(data_loader.dataset)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    features = np.concatenate(features, 0)
    preds = np.concatenate(preds, 0)
    filenames = np.concatenate(filenames, 0)

    if classify:
        # Make a confusion matrix
        try:
            cm = confusion_matrix(y_true=labels, y_pred=preds)
            confusion_matrix_heatmap = make_heatmap(cm, data_loader.dataset.classes)
            writer.add_image(logging_label + '/confusion_matrix', confusion_matrix_heatmap, epoch)
        except ValueError:
            logging.warning('Confusion matrix received weird values')

        # Generate a classification report for each epoch
        logging.info('Classification Report for epoch {}\n'.format(epoch))
        logging.info('\n' + classification_report(y_true=labels,
                                                  y_pred=preds,
                                                  target_names=[str(item) for item in data_loader.dataset.classes]))

    return features, preds, labels, filenames
