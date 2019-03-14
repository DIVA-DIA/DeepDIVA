# Keep the list of models implemented up-2-date
from .CNN_basic import CNN_basic
from .FC_medium import FC_medium
from .FC_simple import FC_simple
from .TNet import TNet
from ._AlexNet import alexnet
from ._ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from .BabyResNet import babyresnet18, babyresnet34, babyresnet50, babyresnet101, babyresnet152
from ._VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from ._Inception_v3 import inception_v3
from ._DenseNet import densenet121, densenet161, densenet169, densenet201
from .CAE_basic import CAE_basic
from .CAE_medium import CAE_medium
from .FusionNet import FusionNet
from .UNet import Unet


"""
Formula to compute the output size of a conv. layer

new_size =  (width - filter + 2padding) / stride + 1
"""
