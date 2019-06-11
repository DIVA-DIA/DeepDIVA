import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

"""
Function which returns the labelled image after applying CRF
adapted from https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf.
"""

def crf(original_image, output, nb_iterations=1, sxy1=(3, 3), sxy2=(80, 80), compat=3, srgb=(13, 13, 13)):
    """

    Parameters explained https://github.com/lucasb-eyer/pydensecrf

    Parameters
    ----------
    original_image : H x W x RGB
         [0:255]
    output : C x H x W
        float confidence of the network


    Returns
    -------
    H x W
        map of the selected labels, [0..C] where C is the number of classes
    """
    original_image = original_image.astype(np.uint8)

    # The output needs to be between 0 and 1
    if np.max(output) > 1 or np.min(output) < 0:
        output = softmax(output, axis=0)

    # Make the array contiguous in memory
    output = output.copy(order='C')

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], output.shape[0])
    U = unary_from_softmax(output)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=sxy1, compat=compat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=sxy2,
                           srgb=srgb,
                           rgbim=original_image,
                           compat=compat*3,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(nb_iterations)

    return np.argmax(Q, axis=0).reshape(original_image.shape[0], original_image.shape[1])




def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p