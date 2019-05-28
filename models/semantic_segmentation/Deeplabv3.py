# Adapted from https://github.com/fregu856/deeplabv3

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import urllib
import os

from models.registry import Model
from models.semantic_segmentation.deeplabv3_resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, \
    ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from models.semantic_segmentation.deeplabv3_aspp import ASPP, ASPP_Bottleneck

CLASS_NAMES = {"deeplabv3_resnet18_os16": ResNet18_OS16,
               "deeplabv3_resnet34_os16": ResNet34_OS16,
               "deeplabv3_resnet50_os16": ResNet50_OS16,
               "deeplabv3_resnet101_os16": ResNet101_OS16,
               "deeplabv3_resnet152_os16": ResNet152_OS16,
               "deeplabv3_resnet18_os8": ResNet18_OS8,
               "deeplabv3_resnet34_os8": ResNet34_OS8,
               }


@Model
def deeplabv3(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3', output_channels, **kwargs)

@Model
def deeplabv3_resnet18_os16(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet18_os16', output_channels, **kwargs)

@Model
def deeplabv3_resnet34_os16(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet34_os16', output_channels, **kwargs)

@Model
def deeplabv3_resnet50_os16(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet50_os16', output_channels, **kwargs)

@Model
def deeplabv3_resnet101_os16(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet101_os16', output_channels, **kwargs)

@Model
def deeplabv3_resnet152_os16(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet152_os16', output_channels, **kwargs)

@Model
def deeplabv3_resnet18_os8(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet18_os8', output_channels, **kwargs)

@Model
def deeplabv3_resnet34_os8(output_channels, **kwargs):
    return deeplabv3_builder('deeplabv3_resnet34_os8', output_channels, **kwargs)

# *********************************************************************************


def deeplabv3_builder(model_name, output_channels, pretrained=False, resume=None, cityscapes=False, **kwargs):
    if model_name=='deeplabv3':
        logging.info('ResNet type not specified, running "deeplabv3_resnet18_os8". (choose from {})'.format(", ".join(CLASS_NAMES.keys())))
        model = DeepLabV3("deeplabv3_resnet18_os8", pretrained, output_channels, **kwargs)
    else:
        model = DeepLabV3(model_name, pretrained, output_channels, **kwargs)

    # load a model from a path
    if resume:
        if os.path.isfile(resume):
            model_dict = torch.load(resume)
            logging.info('Loading a saved model')
            try:
                model.load_state_dict(model_dict['state_dict'], strict=False)
            except Exception as exp:
                logging.warning(exp)
        else:
            logging.error("No model dict found at '{}'".format(resume))

    # load the weights pre-trained on cityscapes dataset (only possible for current "deeplabv3_resnet18_os8" set-up)
    if "deeplabv3_resnet18_os8" and cityscapes:
        try:
            path = get_cityscapes_model_path(**kwargs)
            model.load_state_dict(torch.load(path), strict=False)

        except Exception as exp:
            logging.warning(exp)

    return model


class DeepLabV3(nn.Module):
    def __init__(self, model_name, pretrained, num_classes, **kwargs):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        self.resnet = CLASS_NAMES[model_name](pretrained) # NOTE! specify the type of ResNet here
        if 'resnet18' in model_name or 'resnet34' in model_name:
            self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        else:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = torch.nn.functional.interpolate(output, size=(h, w), mode="bilinear", align_corners=True) # (shape: (batch_size, num_classes, h, w))

        return output


def get_cityscapes_model_path(**kwargs):
    download_path = os.path.join(os.getcwd(), "models/deeplabv3_13_2_2_2_epoch_580.pth")

    if not os.path.exists(download_path):
        url = urllib.parse.urlparse("https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/model_13_2_2_2_epoch_580.pth?raw=true")
        print('Downloading {}...'.format(url.geturl()))
        urllib.request.urlretrieve(url.geturl(), download_path)

    return download_path
