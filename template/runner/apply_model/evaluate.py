# Utils
import logging

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# Torch related stuff
from torch.autograd import Variable
from tqdm import tqdm

from util.misc import save_image_and_log_to_tensorboard
# DeepDIVA
from util.visualization.confusion_matrix_heatmap import make_heatmap


def feature_extract(data_loader, model, writer, epoch, no_cuda, log_interval, classify, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set

    model : torch.nn.module
        The network model being used

    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    epoch : int
        Number of the epoch (for logging purposes)

    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    classify : boolean
        Specifies whether to generate a classification report for the data or not.

    Returns
    -------
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

        data_a = Variable(data, volatile=True)

        # Compute output
        out = model(data_a)

        if multi_crop:
            out = out.view(bs, ncrops, -1).mean(1)

        preds.append([np.argmax(item.data.cpu().numpy()) for item in out])
        features.append(out.data.cpu().numpy())
        labels.append(label)
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
            save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                              image=confusion_matrix_heatmap, global_step=epoch)
        except ValueError:
            logging.warning('Confusion matrix received weird values')

        # Generate a classification report for each epoch
        logging.info('Classification Report for epoch {}\n'.format(epoch))
        logging.info('\n' + classification_report(y_true=labels,
                                                  y_pred=preds,
                                                  target_names=[str(item) for item in data_loader.dataset.classes]))
    else:
        preds = None


    return features, preds, labels, filenames
