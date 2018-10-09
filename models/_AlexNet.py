"""
Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
"""

import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class _AlexNet(nn.Module):
    r"""
    AlexNet model architecture from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
    conv4 : torch.nn.Sequential
    conv5 : torch.nn.Sequential
        Convolutional layers of the network
    fc1 : torch.nn.Linear
    fc2 : torch.nn.Linear
        Fully connected layer
    cl : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=1000, **kwargs):
        """
        Creates an AlexNet model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        """
        super(_AlexNet, self).__init__()

        self.expected_input_size = (227, 227)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_channels),
        )

        # This is an alternative model definition which enables a deeper inspection of the model trough the debugger.
        # It has been developed for didactic purposes and it equivalent to the original definiton.
        # Unfortunately it does not support the imageNet pre-trained loading of the model, therefore is commented.
        #
        # # Convolutional layers
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Dropout2d()
        # )
        #
        # # Fully connected layers
        # self.fc1 = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        # )
        #
        # # Classification layer
        # self.cl = nn.Sequential(
        #     nn.Linear(4096, output_channels),
        # )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

        # This is how we would compute the forward with the alternative model definition.
        # See above for details.
        #
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.cl(x)
        # return x

def alexnet(pretrained=False, **kwargs):
    """
    Returns an AlexNet model, possibly ImageNet pretrained.

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    """
    model = _AlexNet(**kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
