"""
Here are defined the different versions of advanced initialization techniques.
"""

# Utils
import logging
import math

import numpy as np
from sklearn.decomposition import PCA

# Init tools
import util.lda as lda


def perform_lda(index, init_input, init_labels, model, module, module_type):
    # TODO fix this such that only B and W are return values, prolly bring in the assign logic here and leave only the real assign outside

    C = None
    P = None

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

    return B, C, P, W
