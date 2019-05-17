# Adapted from https://github.com/fregu856/deeplabv3

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import urllib
import os

from models.registry import Model
from models.semantic_segmentation.deeplabv3_resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from models.semantic_segmentation.deeplabv3_aspp import ASPP, ASPP_Bottleneck


@Model
def deeplabv3(output_channels, pretrained=False, cityscapes=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DeepLabV3(pretrained, output_channels, **kwargs)

    if cityscapes:
        try:
            path = get_cityscapes_model_path(**kwargs)
            model.load_state_dict(torch.load(path), strict=False)

        except Exception as exp:
            logging.warning(exp)

    return model


class DeepLabV3(nn.Module):
    def __init__(self, pretrained, num_classes, **kwargs):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        # TODO: make different functions for different models
        self.resnet = ResNet18_OS8(pretrained) # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output


def get_cityscapes_model_path(**kwargs):
    download_path = os.path.join(os.getcwd(), "models/deeplabv3_13_2_2_2_epoch_580.pth")

    if not os.path.exists(download_path):
        url = urllib.parse.urlparse("https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/model_13_2_2_2_epoch_580.pth?raw=true")
        print('Downloading {}...'.format(url.geturl()))
        urllib.request.urlretrieve(url.geturl(), download_path)

    return download_path