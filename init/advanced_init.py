"""
Here are defined the different versions of advanced initialization techniques.
"""

# Utils
import logging
import math

import numpy as np
# Torch
import torch
from sklearn.decomposition import PCA

# Init tools
import util.lda as lda


def pure_lda(index, init_input, init_labels, model, module, module_type, **kwargs):
    if index != len(list(model.children())) - 1:
        logging.info('LDA Transform')
        W, B = lda.transform(X=init_input, y=init_labels)
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

    #######################################################################
    # Reshape / Crop the parameters matrix to the proper size
    if 'conv' in module_type:
        # Get kernel size of current layer
        kernel_size = module.kernel_size

        # The T belongs to the reshape operation! It is NOT transposing the input! It is necessary to select columns
        W = W.T.reshape(W.shape[0], module.in_channels, kernel_size[0], kernel_size[1])[:module.out_channels]
        B = B[:module.out_channels]

    #######################################################################
    # Modify the range of the weights (clipping, normalize, ... )
    # W = W / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))
    # B = B / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))

    #######################################################################
    # Check size of W and B
    if module.weight.data.shape[0] > W.shape[0] and (module.weight.data.shape[1] == W.shape[1]):
        """
        This can and will happen when the dimensionality of the data is smaller than the number of neurons. 
        For example on cloud points having more than 2 neurons will trigger this. 
        """
        logging.warning("LDA weight matrix smaller than the number of neurons. Expected {} got {}. "
                        "Filling missing dimensions with default values".format(module.weight.data.shape[0], W.shape[0]))

        W = np.vstack((W, module.weight.data.numpy()[W.shape[0]:, :]))
        B = np.hstack((B, module.bias.data.numpy()[B.shape[0]:]))

    # Keep only necessary dimensions when num_columns > num_desired_dimensions
    if module.weight.data.shape[0] < W.shape[0] and (module.weight.data.shape[1] == W.shape[1]):
        W = W[:module.weight.data.shape[0], :]
        B = B[:module.bias.data.shape[0]]

    return torch.Tensor(W), torch.Tensor(B)


def advanced_lda(index, init_input, init_labels, model, module, module_type, **kwargs):
    if index != len(list(model.children())) - 1:
        logging.info('LDA Transform')
        # Check if size of model allows (has enough neurons)
        if module.weight.data.shape[0] < len(np.unique(init_labels)) * 2:
            logging.error("Model does not have enough neurons. Expected at least |C|*2 got {}".format(module.weight.data.shape[0]))
        W, B = lda.transform(X=init_input, y=init_labels)
    else:
        logging.info('LDA Discriminants')
        W, B = lda.discriminants(X=init_input, y=init_labels)

    #######################################################################
    # Reshape / Crop the parameters matrix to the proper size
    if 'conv' in module_type:
        # Get kernel size of current layer
        kernel_size = module.kernel_size

        # The T belongs to the reshape operation! It is NOT transposing the input! It is necessary to select columns
        W = W.T.reshape(W.shape[0], module.in_channels, kernel_size[0], kernel_size[1])[:module.out_channels]
        B = B[:module.out_channels]

    #######################################################################
    # Modify the range of the weights (clipping, normalize, ... )
    # W = W / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))
    # B = B / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))

    #######################################################################
    # Check size of W and B
    if module.weight.data.shape[0] > W.shape[0] and (module.weight.data.shape[1] == W.shape[1]):
        """
        This can and will happen when the dimensionality of the data is smaller than the number of neurons. 
        For example on cloud points having more than 2 neurons will trigger this. 
        """
        logging.warning("LDA weight matrix smaller than the number of neurons. Expected {} got {}. "
                        "Filling missing dimensions with default values".format(module.weight.data.shape[0], W.shape[0]))

        # Set current values to 0 on the model (all of them)
        # module.weight.data = torch.Tensor(np.zeros(module.weight.data.shape))
        # module.bias.data = torch.Tensor(np.zeros(module.bias.data.shape))

        W = np.vstack((W, module.weight.data.numpy()[W.shape[0]:, :]))
        B = -np.matmul(W, B)
        B = np.hstack((B, module.bias.data.numpy()[B.shape[0]:]))

    # Keep only necessary dimensions when num_columns > num_desired_dimensions
    if module.weight.data.shape[0] < W.shape[0] and (module.weight.data.shape[1] == W.shape[1]):
        W = W[:module.weight.data.shape[0], :]
        B = B[:module.bias.data.shape[0]]

    return torch.Tensor(W), torch.Tensor(B)


def lda_pca(index, init_input, init_labels, model, module, module_type, **kwargs):
    num_classes = np.unique(init_labels)

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
    if 'conv' in module_type:
        # Get kernel size of current layer
        kernel_size = module.kernel_size

        # The T belongs to the reshape operation! It is NOT transposing the input! It is necessary to select columns
        W = W.T.reshape(W.shape[0], module.in_channels, kernel_size[0], kernel_size[1])[:module.out_channels]
        B = B[:module.out_channels]

        P = P.T.reshape(P.shape[0], module.in_channels, kernel_size[0], kernel_size[1])[:module.out_channels]

    else:
        W = W / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))
        B = B / (max(np.max(np.abs(B)), np.max(np.abs(W))) * math.sqrt(W.shape[0]))

    if 'conv' in module_type:
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

        n = int(np.max([num_classes, np.round(B.shape[0] / 2)]))
        lp_ratio = (np.abs(W.max()) + np.abs(W.min())) / (np.abs(P.max()) + np.abs(P.min()))
        module.weight.data[0:n] += torch.Tensor(W)[0:n]
        module.bias.data[0:n] += torch.Tensor(B)[0:n]
        module.weight.data[n:] += lp_ratio * torch.Tensor(P)[0:-n]
        module.bias.data[n:] += lp_ratio * torch.Tensor(C)[0:B.shape[0] - n]

    return W, B
