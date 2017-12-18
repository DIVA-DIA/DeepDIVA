# Keep the list of models implemented up-2-date
from .AlexNet import AlexNet, alexnet
from .CNN_basic import CNN_basic
from .LDA_test import LDA_simple, LDA_CIFAR

"""
Formula to compute the output size of a conv. layer

new_size =  (width - filter + 2padding) / stride + 1
"""
