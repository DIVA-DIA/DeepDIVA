"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import logging
import math

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d

# Init tools
import util.lda as lda


# TODO two parameters should be A() and B() where A is used to init everything and B only the last layer.
def init_model(model, data_loader, *args, **kwargs):
    """
    Initialize a standard CNN composed by convolutional layer followed by fully
    connected layers.
    :param model:
        the network to initialize
    :param data:
        the dataloader to take the data from
    :param args:
        parameters for the function. In particular:
            > num_points: integer
            specifies how many points should be used to compute the data-driven
            initialization
    :param kwargs:
        parameters for the function.
    :return:
        nothing, the parameters of the network are modified in place
    """

    # Collect initial data
    logging.debug('Collect initial data')
    X = []
    y = []
    for i, (input, target) in enumerate(data_loader, 1):
        X.append(torch.autograd.Variable(input))
        y.append(torch.autograd.Variable(target))
        if i * data_loader.batch_size >= kwargs['num_points']:
            break

    ###############################################################################################
    # Iterate over all layers
    logging.info('Iterate over all layers')
    for index, module in enumerate(model.children()):
        logging.info('Layer: {}'.format(index + 1))

        #######################################################################
        # In case of conv layer, get the patches of data
        init_input, init_labels = None, None
        if 'conv' in str(type(list(module.children())[0])):
            logging.info('Get the patches of data')
            # Get kernel size of current layer
            kernel_size = module[0].kernel_size
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
        if index != len(list(model.children())) - 1:
            logging.info('LDA Transform')
            W, B = lda.transform(X=init_input, y=init_labels)
            pca = PCA().fit(init_input)
            P = pca.components_.T  # Don't even think about touching this T!
            C = pca.mean_

        else:
            logging.info('LDA Discriminants')
            W, B = lda.discriminants(X=init_input, y=init_labels)

        #######################################################################
        # Reshape / Crop the parameters matrix to the proper size
        if 'conv' in str(type(list(module.children())[0])):
            # The T belongs to the reshape operation! It is NOT transposing the input! It is necessary to select columns
            W = W.T.reshape(W.shape[0], module[0].in_channels, kernel_size[0], kernel_size[1])[:module[0].out_channels]
            B = B[:module[0].out_channels]

            P = P.T.reshape(P.shape[0], module[0].in_channels, kernel_size[0], kernel_size[1])[:module[0].out_channels]

        else:
            W = W / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))
            B = B / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))

        # Assign parameters
        logging.info('Assign parameters')
        if 'conv' in str(type(list(module.children())[0])):
            # if False:
            # TODO un-hard-code the 10 as number of classes
            """
            module[0].weight.data[0:10] = torch.Tensor(W)[0:10]
            module[0].bias.data[0:10] = torch.Tensor(B)[0:10]
            module[0].weight.data[10:] = torch.Tensor(P)[0:-10]
            module[0].bias.data[10:] = torch.Tensor(C)[0:B.shape[0]-10]
            """
            # WITH NOISE
            ns_ratio = (np.abs(W.max()) + np.abs(W.min())) / (
            np.abs(module[0].weight.data.max()) + np.abs(module[0].weight.data.min()))

            module[0].weight.data *= (ns_ratio / 3)
            module[0].bias.data *= (ns_ratio / 3)

            n = int(np.max([10, np.round(B.shape[0] / 2)]))
            lp_ratio = (np.abs(W.max()) + np.abs(W.min())) / (np.abs(P.max()) + np.abs(P.min()))
            module[0].weight.data[0:n] += torch.Tensor(W)[0:n]
            module[0].bias.data[0:n] += torch.Tensor(B)[0:n]
            module[0].weight.data[n:] += lp_ratio * torch.Tensor(P)[0:-n]
            module[0].bias.data[n:] += lp_ratio * torch.Tensor(C)[0:B.shape[0] - n]

        else:
            module[0].weight.data = torch.Tensor(W)
            module[0].bias.data = torch.Tensor(B)

        # If the layer is not convolutional then flatten the data because
        # we assume it is a fully connected one
        if 'conv' not in str(type(list(model.children())[index][0])) and 'conv' in str(
                type(list(model.children())[index - 1][0])):
            logging.info('Flattening input')
            for i, minibatch in enumerate(X):
                X[i] = X[i].view(X[i].size(0), -1)

        # Forward pass
        logging.info('Forward pass')
        for i, minibatch in enumerate(X):
            X[i] = module(X[i])


def get_patches(X, y, kernel_size, max_patches=.9):
    """
    Extract patches out of a set of N images passed as parameter. Additionally returns the relative set of labels
    corresponding to each patch
    :param X: ndarray(N,depth,width,height)
        set of images to take the patch from.
    :param y: ndarray(N,)
        set of labels (GT). There must be one for each of the images contained in X.
    :param kernel_size: tuple(width,height)
        size of the kernel to use to extract the patches.
    :return:
    all_patches: ndarray(N*patch_per_image,depth*width*height)
        list of patches flattened
    labels: ndarray(N*patch_per_image,)
        list of labels for each of the elements of 'all_patches'
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

