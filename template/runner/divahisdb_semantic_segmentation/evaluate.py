# Utils
import logging
import os
import time

import matplotlib
import numpy as np
from PIL import Image
# Torch related stuff
from tqdm import tqdm

from datasets.custom_transform_library.functional import gt_to_one_hot_hisdb as gt_to_one_hot
from util.evaluation.metrics.accuracy import accuracy_segmentation
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, make_colour_legend_image
from util.visualization.DIVAHisDB_layout_analysis_tool_visualization import generate_layout_analysis_output
from .setup import output_to_class_encodings


def validate(val_loader, model, criterion, writer, epoch, class_encodings, no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------

    val_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    class_encodings : List
        Contains the classes (range of ints)
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    # 'Run' is injected in kwargs at runtime IFF it is a multi-run event
    multi_run = kwargs['run'] if 'run' in kwargs else None

    num_classes = len(class_encodings)

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Compute output
        output = model(input)

        # Compute and record the loss
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # Compute and record the accuracy
        _, _, mean_iu_batch, _ = accuracy_segmentation(target.cpu().numpy(), get_argmax(output), num_classes)
        meanIU.update(mean_iu_batch, input.size(0))

        # Add loss and meanIU to Tensorboard
        scalar_label = 'val/mb_loss' if multi_run is None else 'val/mb_loss_{}'.format(multi_run)
        writer.add_scalar(scalar_label, loss.item(), epoch * len(val_loader) + batch_idx)
        scalar_label = 'val/mb_meanIU' if multi_run is None else 'val/mb_meanIU_{}'.format(multi_run)
        writer.add_scalar(scalar_label, mean_iu_batch, epoch * len(val_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description('val epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(val_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))


    # Logging the epoch-wise meanIU
    scalar_label = 'val/mb_meanIU' if multi_run is None else 'val/mb_meanIU_{}'.format(multi_run)
    writer.add_scalar(scalar_label, meanIU.avg, epoch)

    logging.info(_prettyprint_logging_label("val") +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    return meanIU.avg


def test(test_loader, model, criterion, writer, epoch, class_encodings, img_names_sizes_dict, dataset_folder,
         no_cuda=False, log_interval=10, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    img_names_sizes_dict: dictionary {str: (int, int)}
        Key: gt image name (with extension), Value: image size
    test_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    class_encodings : List
        Contains the range of encoded classes
    img_names_sizes_dict
        # TODO
    dataset_folder : str
        # TODO
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    """
    # 'Run' is injected in kwargs at runtime IFF it is a multi-run event
    multi_run = kwargs['run'] if 'run' in kwargs else None

    num_classes = len(class_encodings)

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    meanIU = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Need to store the images currently being processes
    canvas = {}

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:
        # Unpack input
        input, top_left_coordinates, test_img_names = input

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Compute output
        output = model(input)

        # Compute and record the loss
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # Compute and record the batch meanIU
        _, _, mean_iu_batch, _ = accuracy_segmentation(target.cpu().numpy(), get_argmax(output), num_classes)

        # Add loss and meanIU to Tensorboard
        scalar_label = 'test/mb_loss' if multi_run is None else 'test/mb_loss_{}'.format(multi_run)
        writer.add_scalar(scalar_label, loss.item(), epoch * len(test_loader) + batch_idx)
        scalar_label = 'test/mb_meanIU' if multi_run is None else 'test/mb_meanIU_{}'.format(multi_run)
        writer.add_scalar(scalar_label, mean_iu_batch, epoch * len(test_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description('test epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(test_loader)))
            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             meanIU='{meanIU.avg:.3f}\t'.format(meanIU=meanIU),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

        # Output needs to be patched together to form the complete output of the full image
        # patches are returned as a sliding window over the full image, overlapping sections are averaged
        for patch, x, y, img_name in zip(output.data.cpu().numpy(), top_left_coordinates[0].numpy(), top_left_coordinates[1].numpy(), test_img_names):

            # Is a new image?
            if not img_name in canvas:
                # Create a new image of the right size filled with NaNs
                canvas[img_name] = np.empty((num_classes, *img_names_sizes_dict[img_name]))
                canvas[img_name].fill(np.nan)

            # Add the patch to the image
            canvas[img_name] = merge_patches(patch, (x, y), canvas[img_name])

            # Save the image when done
            if not np.isnan(np.sum(canvas[img_name])):
                # Save the final image
                mean_iu = process_full_image(img_name, canvas[img_name], multi_run, dataset_folder, class_encodings)
                # Update the meanIU
                meanIU.update(mean_iu, 1)
                # Remove the entry
                canvas.pop(img_name)
                logging.info("\nProcessed image {} with mean IU={}".format(img_name, mean_iu))

    # Canvas MUST be empty or something was wrong with coverage of all images
    assert len(canvas) == 0

    # Logging the epoch-wise meanIU
    scalar_label = 'test/mb_meanIU' if multi_run is None else 'test/mb_meanIU_{}'.format(multi_run)
    writer.add_scalar(scalar_label, meanIU.avg, epoch)

    logging.info(_prettyprint_logging_label("test") +
                 ' epoch[{}]: '
                 'MeanIU={meanIU.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, meanIU=meanIU))

    return meanIU.avg


def get_argmax(output):
    """ Gets the argmax values for each sample in the minibatch"""
    # TODO check with Vinay & Michele if correct
    return np.array([np.argmax(o, axis=0) for o in output.data.cpu().numpy()])



def merge_patches(patch, coordinates, full_output):
    """
    This function merges the patch into the full output image
    Overlapping values are resolved by taking the max.

    Parameters
    ----------
    patch: numpy matrix of size [batch size x #C x crop_size x crop_size]
        a patch from the larger image
    coordinates: tuple of ints
        top left coordinates of the patch within the larger image for all patches in a batch
    full_output: numpy matrix of size [#C x H x W]
        output image at full size
    Returns
    -------
    full_output: numpy matrix [#C x Htot x Wtot]
    """
    assert len(full_output.shape) == 3
    assert full_output.size != 0

    # Resolve patch coordinates
    x1, y1 = coordinates
    x2, y2 = x1 + patch.shape[1], y1 + patch.shape[2]

    # If this triggers it means that a patch is 'out-of-bounds' of the image and that should never happen!
    assert x2 <= full_output.shape[1]
    assert y2 <= full_output.shape[2]

    mask = np.isnan(full_output[:, x1:x2, y1:y2])
    # if still NaN in full_output just insert value from crop, if there is a value then take max
    full_output[:, x1:x2, y1:y2] = np.where(mask, patch, np.maximum(patch, full_output[:, x1:x2, y1:y2]))

    return full_output


def process_full_image(image_name, output, multi_run, dataset_folder, class_encodings):
    """
    Helper function to save the output during testing

    Parameters
    ----------
    meanIU.avg : float
        MeanIU of the model of the evaluated split
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    image_name: str
        name of the image that is saved
    output: numpy matrix of size [#C x H x W]
        output image at full size
    dataset_folder: str
        path to the dataset folder

    Returns
    -------
    mean_iu : float
        mean iu of this image
    """
    num_classes = len(class_encodings)

    # Load GT
    with open(os.path.join(dataset_folder, "test", "gt", image_name), 'rb') as f:
        with Image.open(f) as img:
            #ground_truth = np.rollaxis(np.array(img), 1, 0)
            ground_truth = np.array(img)

    # Get the ground truth mapping
    target = np.argmax(gt_to_one_hot(ground_truth, class_encodings, True).numpy(), axis=0)

    # Get boundary pixels
    boundary_mask = ground_truth[:, :, 0].astype(np.uint8) == 128

    # Get predictions and filter their values for the boundary pixels
    prediction = np.argmax(output, axis=0)
    prediction[boundary_mask] = target[boundary_mask]

    # Adjust the gt_image for the border pixel -> set to background (1)
    ground_truth[boundary_mask] = 1

    # Compute and record the meanIU of the whole image
    _, _, mean_iu, _ = accuracy_segmentation(target, prediction, num_classes)

    output_encoded = output_to_class_encodings(output, class_encodings)
    scalar_label = 'output_{}'.format(image_name) if multi_run is None else 'output_{}_{}'.format(multi_run, image_name)
    _save_output_evaluation(class_encodings, output_encoded=output_encoded, gt_image=ground_truth, tag=scalar_label, multi_run=multi_run)

    return mean_iu


def _save_output_evaluation(class_encodings, output_encoded, gt_image, tag, multi_run=None):
    """Utility function to save image in the output folder and also log it to Tensorboard.

    Parameters
    ----------
    class_encodings : List
        Contains the range of encoded classes
    tag : str
        Name of the image.
    output_encoded : ndarray [W x H x C] in RGB
        Image to be saved
    gt_image : ndarray [W x H x C] in RGB
        Image to be saved
    multi_run : int
        Epoch/Mini-batch counter.

    Returns
    -------
    None

    """
    # ##################################################################################################################
    # 1. Create true output

    # Get output folder using the FileHandler from the logger.
    # (Assumes the file handler is the last one)
    output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)
    dest_filename = os.path.join(output_folder, 'images', "output", tag  if multi_run is None else tag + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    # Save the output
    Image.fromarray(output_encoded.astype(np.uint8)).save(dest_filename)

    # ##################################################################################################################
    # 2. Make a more human readable output -> one colour per class
    tag_col = "coloured/" + tag

    dest_filename = os.path.join(output_folder, 'images', tag_col if multi_run is None else tag_col + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(output_encoded)
    blue = output_encoded[:, :, 2]  # Extract just blue channel

    # Colours are in RGB
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]

    # Get the mask for each colour
    masks = {color: (blue == i) > 0 for color, i in zip(colors, class_encodings)}

    # Color the image with relative colors
    for color, mask in masks.items():
        img[mask] = color

    # Make and save the class color encoding
    color_encoding = {str(i): color for color, i in zip(colors, class_encodings)}

    make_colour_legend_image(os.path.join(os.path.dirname(dest_filename), "output_visualizations_colour_legend.png"),
                             color_encoding)

    # Write image to output folder
    Image.fromarray(img.astype(np.uint8)).save(dest_filename)

    # ##################################################################################################################
    # 3. Layout analysis evaluation
    # Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator

    img_la = np.copy(output_encoded)
    tag_la = "layout_analysis_evaluation/" + tag

    dest_filename = os.path.join(output_folder, 'images', tag_la if multi_run is None else tag_la + '_{}'.format(multi_run))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    generate_layout_analysis_output(os.path.join(output_folder, 'images'), gt_image, img_la, dest_filename, legend=True)


