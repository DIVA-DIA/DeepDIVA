"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import logging

import numpy as np
import torch
from sklearn.feature_extraction.image import extract_patches_2d

# DeepDIVA
from init.advanced_init import perform_lda


def init_model(model, data_loader, num_samples, **kwargs):
    """
    Initialize a standard CNN composed by convolutional layer followed by fully
    connected layers.

    Parameters:
    -----------
    :param model:
        the network to initialize

    :param data_loader:
        the dataloader to take the data from

    :param num_samples: integer
        specifies how many points should be used to compute the data-driven
        initialization

    :param kwargs:
        parameters for the function.

    :return:
        nothing, the parameters of the network are modified in place
    """

    # Collect initial data
    logging.debug('Collect initial data')
    X, y = _collect_initial_data(data_loader, num_samples)

    ###############################################################################################
    # Iterate over all layers
    logging.info('Iterate over all layers')
    for index, layer in enumerate(model.children()):

        # Get module from layer
        module_type, module = get_module_from_layer(layer)
        logging.info('Layer: {} - layer: {}'.format(index + 1, module_type))

        #######################################################################
        # In case of conv layer, get the patches of data, else just reshape minibatches into a matrix LDA friendly
        if 'conv' in module_type:
            logging.info('Get the patches of kernel size {} from data'.format(kernel_size))
            # Get kernel size of current layer
            kernel_size = module.kernel_size
            # Get the patches
            patches, labels = get_patches(
                np.array([element.data.numpy() for minibatch in X for element in minibatch]),
                np.squeeze(minibatches_to_matrix(y)),
                kernel_size=kernel_size
            )
            init_input = patches.reshape(patches.shape[0], -1)
            init_labels = labels
        else:
            init_input = minibatches_to_matrix(X)
            init_labels = np.squeeze(minibatches_to_matrix(y))

        #######################################################################
        # Compute data-driven parameters
        logging.info('Compute data-driven parameters')
        B, C, P, W = perform_lda(index, init_input, init_labels, model, module, module_type)

        #######################################################################
        # Assign parameters
        logging.info('Assign parameters')
        if 'conv' in module_type:
            # if False:
            # TODO un-hard-code the 10 as number of classes
            """
            module.weight.data[0:10] = torch.Tensor(W)[0:10]
            module.bias.data[0:10] = torch.Tensor(B)[0:10]
            module.weight.data[10:] = torch.Tensor(P)[0:-10]
            module.bias.data[10:] = torch.Tensor(C)[0:B.shape[0]-10]
            """
            # WITH NOISE
            ns_ratio = (np.abs(W.max()) + np.abs(W.min())) / (
            np.abs(module.weight.data.max()) + np.abs(module.weight.data.min()))

            module.weight.data *= (ns_ratio / 3)
            module.bias.data *= (ns_ratio / 3)

            n = int(np.max([10, np.round(B.shape[0] / 2)]))
            lp_ratio = (np.abs(W.max()) + np.abs(W.min())) / (np.abs(P.max()) + np.abs(P.min()))
            module.weight.data[0:n] += torch.Tensor(W)[0:n]
            module.bias.data[0:n] += torch.Tensor(B)[0:n]
            module.weight.data[n:] += lp_ratio * torch.Tensor(P)[0:-n]
            module.bias.data[n:] += lp_ratio * torch.Tensor(C)[0:B.shape[0] - n]

        else:
            module.weight.data = torch.Tensor(W)
            module.bias.data = torch.Tensor(B)

        # UPDATE: as of the introduction of Flatten() module, this is no longer necessary.
        # If the layer is not convolutional then flatten the data because we assume it is a fully connected one
        # if 'conv' not in str(type(list(model.children())[index][0])) and 'conv' in str(
        #         type(list(model.children())[index - 1][0])):
        #     logging.info('Flattening input')
        #     for i, minibatch in enumerate(X):
        #         X[i] = X[i].view(X[i].size(0), -1)

        # Forward pass
        logging.info('Forward pass')
        for i, _ in enumerate(X):
            X[i] = layer(X[i])


def _collect_initial_data(data_loader, num_points):
    X = []
    y = []
    for i, (input, target) in enumerate(data_loader, 1):
        X.append(torch.autograd.Variable(input))
        y.append(torch.autograd.Variable(target))
        if i * data_loader.batch_size >= num_points:
            break

    return X, y


def minibatches_to_matrix(X):
    """
    Flattens the a list of matrices of shape[[minibatch, dim_1, ..., dim_n], [minibatch, dim_1, ..., dim_n] ...] such
    that it becomes [minibatch * len(list), dim_1 * dim_2 ... *dim_n]
    :param X:
        list of matrices
    :return:
        flattened matrix
    """
    return np.array([sample.data.view(-1).numpy() for minibatch in X for sample in minibatch])


def get_module_from_layer(layer):
    # Skip Flatten() modules at the beginning of the layer
    if 'Flatten' in str(type(list(layer.children())[0])):
        module_type = str(type(list(layer.children())[1]))
        module = list(layer.children())[1]
    else:
        module_type = str(type(list(layer.children())[0]))
        module = list(layer.children())[0]

    return module_type, module


def get_patches(X, y, kernel_size, max_patches=.9):
    """
    Extract patches out of a set of N images passed as parameter. Additionally returns the relative set of labels
    corresponding to each patch

    Parameters:
    -----------
    :param X: ndarray(N,depth,width,height)
        set of images to take the patch from.

    :param y: ndarray(N,)
        set of labels (GT). There must be one for each of the images contained in X.

    :param kernel_size: tuple(width,height)
        size of the kernel to use to extract the patches.

    :param max_patches : int or double [0:0.99]
        number of patches to extract. Exact if int, ratio of all possible patches if double.

    :return: all_patches: ndarray(N*patch_per_image,depth*width*height), labels: ndarray(N*patch_per_image,)
        list of patches flattened and list of labels for each of the elements of 'all_patches'
    """
    # Init the return values
    all_patches, labels = [], []

    # For all images in X
    for image, label in zip(X, y):
        # Transform the image in the right format for extract_patches_2d(). Needed as channels are not in same order
        image = np.transpose(image, axes=[1, 2, 0])
        # Extract the patches
        patches = extract_patches_2d(image, kernel_size, max_patches=max_patches)
        # Append the patches to the list of all patches extracted and "restore" the order of the channels.
        all_patches.append(np.transpose(patches, axes=[0, 3, 1, 2]))
        # Append the labels to the list of labels, by replicating the current one for as many patches has been extracted
        labels.append(np.repeat(label, len(patches)))

    # Flatten everything (here 'all_patches' is a list of lists in which each element is a 3D  patch )
    all_patches = np.array([sample for minibatch in all_patches for sample in minibatch])
    labels = np.array([sample for minibatch in labels for sample in minibatch])
    return all_patches, labels


