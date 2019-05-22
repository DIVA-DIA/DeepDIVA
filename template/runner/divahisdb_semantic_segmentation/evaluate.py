# Utils
import logging
import time
import warnings
import os
import numpy as np
from PIL import Image
import colorsys
import matplotlib

# Torch related stuff
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard, get_distinct_colors, \
    make_colour_legend_image
from util.visualization.confusion_matrix_heatmap import make_heatmap
from util.visualization.DIVAHisDB_layout_analysis_tool_visualization import generate_layout_analysis_output
from util.evaluation.metrics.accuracy import accuracy_segmentation
from template.setup import _load_class_frequencies_weights_from_file
from .setup import one_hot_to_np_rgb, one_hot_to_full_output
from datasets.custom_transform_library.functional import gt_to_one_hot_hisdb as gt_to_one_hot


def validate(data_loader, model, criterion, writer, epoch, class_encodings, no_val_conf_matrix, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    class_encodings : list
        Contains the classes (range of ints)
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    logging_label = "val"

    num_classes = len(class_encodings)
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # Compute and record the accuracy TODO check with Vinay & Michele if correct
        acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target_var.cpu().numpy(), output_argmax, num_classes)
        meanIU.update(mean_iu, input.size(0))

        # Get the predictions
        _ = [preds.append(item) for item in output_argmax]
        _ = [targets.append(item) for item in target_var.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        try:
            log_loss = loss.item()
        except AttributeError:
            log_loss = loss.data[0]

        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', log_loss, epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU', mean_iu, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), log_loss,
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU_{}'.format(multi_run), mean_iu,
                               epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Make a confusion matrix
    if not no_val_conf_matrix:
        try:
            # targets_flat = np.array(targets).flatten()
            # preds_flat = np.array(preds).flatten()
            # calculate confusion matrices
            cm = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)])
            confusion_matrix_heatmap = make_heatmap(cm, [str(i) for i in class_encodings])

            # load the weights
            # weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers, runner_class)
            # sample_weight = [weights[i] for i in np.array(targets).flatten()]
            # cm_w = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)],
            #                         sample_weight=[weights[i] for i in np.array(targets).flatten()])
            # confusion_matrix_heatmap_w = make_heatmap(np.round(cm_w*100).astype(np.int), class_names)
        except ValueError:
            logging.warning('Confusion Matrix did not work as expected')
            confusion_matrix_heatmap = np.zeros((10, 10, 3))
            # confusion_matrix_heatmap_w = confusion_matrix_heatmap
    else:
        logging.info("No confusion matrix created.")

    # Logging the epoch-wise accuracy and saving the confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/meanIU', meanIU.avg, epoch)
        if not no_val_conf_matrix :
            save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                              image=confusion_matrix_heatmap, global_step=epoch)
            # save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted',
            #                                   image=confusion_matrix_heatmap_w, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), meanIU.avg, epoch)
        if not no_val_conf_matrix:
            save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                              image = confusion_matrix_heatmap, global_step = epoch)
            # save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted{}'.format(multi_run),
            #                                   image=confusion_matrix_heatmap_w, global_step=epoch)


    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    return meanIU.avg


def test(data_loader, model, criterion, writer, epoch, class_encodings, img_names_sizes_dict, dataset_folder, inmem,
         workers, runner_class, use_boundary_pixel, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    class_encodings : list [int]
        Contains the range of encoded classes
    img_names_sizes_dict: dictionary {str: (int, int)}
        Key: gt image name (with extension), Value: image size
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    logging_label = "test"

    num_classes = len(class_encodings)
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    # needed for test phase output generation
    current_img_name = ""

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        input, top_left_coordinates, test_img_names = input

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        output_argmax = np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # Compute and record the batch meanIU TODO check with Vinay & Michele if correct
        acc_batch, acc_cls_batch, mean_iu_batch, fwavacc_batch = accuracy_segmentation(target_var.cpu().numpy(), output_argmax, num_classes)

        # Add loss and accuracy to Tensorboard
        try:
            log_loss = loss.item()
        except AttributeError:
            log_loss = loss.data[0]

        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', log_loss, epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU', mean_iu_batch, epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), log_loss,
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_meanIU_{}'.format(multi_run), mean_iu_batch,
                               epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

        # Output needs to be patched together to form the complete output of the full image
        # patches are returned as a sliding window over the full image, overlapping sections are averaged
        one_hots = output.data.cpu().numpy()
        for one_hot, x, y, img_name in zip(one_hots, top_left_coordinates[0].numpy(), top_left_coordinates[1].numpy(), test_img_names):
            # new image
            if img_name != current_img_name:
                if len(current_img_name) > 0:
                    # save the old one before starting the new one
                    pred, target, mean_iu = _save_test_img_output(current_img_name, combined_one_hot, multi_run, dataset_folder,
                                                                  logging_label, writer, epoch, class_encodings, use_boundary_pixel)
                    preds.append(pred)
                    targets.append(target)
                    # update the meanIU
                    meanIU.update(mean_iu, 1)

                # start the combination of the new image
                logging.info("Starting segmentation of image {}".format(img_name))
                combined_one_hot = []
                current_img_name = img_name

            combined_one_hot = one_hot_to_full_output(one_hot, (x, y), combined_one_hot, img_names_sizes_dict[img_name])

    # save the final image
    pred, target, mean_iu = _save_test_img_output(current_img_name, combined_one_hot, multi_run, dataset_folder, logging_label, writer, epoch, class_encodings, use_boundary_pixel)
    preds.append(pred)
    targets.append(target)
    # update the meanIU
    meanIU.update(mean_iu, 1)

    # Make a confusion matrix
    try:
        #targets_flat = np.array(targets).flatten()
        #preds_flat = np.array(preds).flatten()
        # load the weights
        weights = _load_class_frequencies_weights_from_file(dataset_folder, inmem, workers, runner_class)
        # calculate the confusion matrix
        cm = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(), labels=[i for i in range(num_classes)])
        cm_w = confusion_matrix(y_true=np.array(targets).flatten(), y_pred=np.array(preds).flatten(),
                                labels=[i for i in range(num_classes)], sample_weight=[weights[i] for i in np.array(targets).flatten()])
        confusion_matrix_heatmap = make_heatmap(cm, [str(i) for i in class_encodings])
        confusion_matrix_heatmap_w = make_heatmap(np.round(cm_w*100).astype(np.int), [str(i) for i in class_encodings])

    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')
        confusion_matrix_heatmap = np.zeros((10, 10, 3))
        confusion_matrix_heatmap_w = confusion_matrix_heatmap

    # Logging the epoch-wise accuracy and saving the confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/meanIU', meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image=confusion_matrix_heatmap, global_step=epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted',
                                          image=confusion_matrix_heatmap_w, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), meanIU.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                          image=confusion_matrix_heatmap, global_step=epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_weighted{}'.format(multi_run),
                                          image=confusion_matrix_heatmap_w, global_step=epoch)


    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    return meanIU.avg


def _save_test_img_output(img_to_save, one_hot, multi_run, dataset_folder, logging_label, writer, epoch, class_encodings,
                          use_boundary_pixel):
    """
    Helper function to save the output during testing

    Parameters
    ----------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    img_to_save: str
        name of the image that is saved
    one_hot: numpy array
        one hot encoded output of the network for the whole image
    dataset_folder: str
        path to the dataset folder

    Returns
    -------
    pred, target: numpy arrays
        argmax of the predicted and target values for the image
    """
    num_classes = len(class_encodings)

    logging.info("Finished segmentation of image {}. Saving output...".format(img_to_save))
    np_rgb = one_hot_to_np_rgb(one_hot, class_encodings)
    # add full image to predictions
    pred = np.argmax(one_hot, axis=0)
    # open full ground truth image
    gt_img_path = os.path.join(dataset_folder, logging_label, "gt", img_to_save)

    with open(gt_img_path, 'rb') as f:
        with Image.open(f) as img:
            ground_truth = np.array(img.convert('RGB'))

    # get the ground truth mapping
    target = np.argmax(gt_to_one_hot(ground_truth, class_encodings, use_boundary_pixel).numpy(), axis=0)

    # border pixels can be classified as background or foreground -> set to the same in pred and target
    if use_boundary_pixel:
        border_mask = ground_truth[:, :, 0].astype(np.uint8) != 0
        pred[border_mask] = target[border_mask]
        # adjust the gt_image for the border pixel -> set to background (1)
        ground_truth[border_mask] = 1

    # Compute and record the meanIU of the whole image TODO check with Vinay & Michele if correct
    acc, acc_cls, mean_iu, fwavacc = accuracy_segmentation(target, pred, num_classes)
    txt = " (adjusted for the boundary pixel)" if use_boundary_pixel else ""
    logging.info("MeanIU {}: {}{}".format(img_to_save, mean_iu, txt))

    if multi_run is None:
        # writer.add_scalar(logging_label + '/meanIU', mean_iu, epoch)
        save_image_and_log_to_tensorboard_segmentation(class_encodings, writer, tag=logging_label + '/output_{}'.format(img_to_save),
                                                       image=np_rgb,
                                                       gt_image=ground_truth)
    else:
        # writer.add_scalar(logging_label + '/meanIU_{}'.format(multi_run), mean_iu, epoch)
        save_image_and_log_to_tensorboard_segmentation(class_encodings, writer, tag=logging_label + '/output_{}_{}'.format(multi_run,
                                                                                                          img_to_save),
                                                       image=np_rgb,
                                                       gt_image=ground_truth)

    return pred, target, mean_iu


def save_image_and_log_to_tensorboard_segmentation(class_encodings, writer=None, tag=None, image=None, global_step=None, gt_image=None):
    """Utility function to save image in the output folder and also log it to Tensorboard.
    ALL IMAGES ARE IN RGB FORMAT
    Parameters
    ----------
    writer : tensorboardX.writer.SummaryWriter object
        The writer object for Tensorboard
    tag : str
        Name of the image.
    image : ndarray [W x H x C] in RGB
        Image to be saved and logged to Tensorboard.
    global_step : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """
    # 1. Create true output

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag + '_{}'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag)

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    # save the output
    result = Image.fromarray(image.astype(np.uint8))
    result.save(dest_filename)

    # 2. Make a more human readable output -> one colour per class
    tag_col = "coloured_" + tag

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

    if global_step is not None:
        dest_filename = os.path.join(output_folder, 'images', tag_col + '_{}'.format(global_step))
    else:
        dest_filename = os.path.join(output_folder, 'images', tag_col)

    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(image)
    blue = image[:, :, 2]  # Extract just blue channel

    # Colours are in RGB
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]
    #colors = get_distinct_colors(len(class_encodings))

    # get the mask for each colour
    masks = {color: (blue == i) > 0 for color, i in zip(colors, class_encodings)}

    for color, mask in masks.items():
        img[mask] = color

    # make and save the class color encoding
    color_encoding = {str(i): color for color, i in zip(colors, class_encodings)}
    if gt_image is not None:
        color_encoding["classified wrong"] = (0, 0, 0)

    make_colour_legend_image(os.path.join(os.path.dirname(dest_filename), "output_visualizations_colour_legend.png"),
                             color_encoding)

    # Write image to output folder
    result = Image.fromarray(img.astype(np.uint8))
    result.save(dest_filename)

    # 3. Make a visualization that highlights the wrongfully classified pixels
    if gt_image is not None:
        tag_col = "errors_coloured_" + tag
        img_overlay = np.copy(img)

        dest_filename = os.path.join(output_folder, 'images', tag_col)

        if not os.path.exists(os.path.dirname(dest_filename)):
            os.makedirs(os.path.dirname(dest_filename))

        # set the wrongfully classified pixels to black
        correct_mask = blue != gt_image[:, :, 2]
        img_overlay[correct_mask] = (0, 0, 0)

        result = Image.fromarray(img_overlay.astype(np.uint8))
        result.save(dest_filename)

    # 4. Layout analysis evaluation
    # Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator
    if gt_image is not None:
        img_la = np.copy(image)
        tag_la = "layout_analysis_evaluation_" + tag
        # Get output folder using the FileHandler from the logger.
        # (Assumes the file handler is the last one)
        output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

        if global_step is not None:
            dest_filename = tag_la + '_{}'.format(global_step)
        else:
            dest_filename = tag_la

        generate_layout_analysis_output(os.path.join(output_folder, 'images'), gt_image, img_la, dest_filename, legend=True)

    return
