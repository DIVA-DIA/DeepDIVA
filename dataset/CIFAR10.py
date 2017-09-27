"""
This is class taking care of CIFAR-10 dataset

Example of initialization:
cifar = CIFAR10("train","./data/dataset/", torchvision.transforms.Compose([torchvision.transforms.Normalize]))
"""

import torch.utils.data.Dataset as dataset


class CIFAR10(dataset):
    def __init__(self, type, path="./data/datasets", transforms=None, *arg, **kw):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
