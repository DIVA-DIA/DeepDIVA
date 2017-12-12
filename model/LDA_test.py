"""
CNN with 6 conv layers and 3 fully connected classification layer
Designed for CIFAR (input: 32x32x3)
"""

import torch.nn as nn


class LDA_CIFAR(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
    :var conv4   : torch.nn.Conv2d
    :var conv5   : torch.nn.Conv2d
    :var conv6   : torch.nn.Conv2d
        The first six convolutional layers of the network
    :var fc      : torch.nn.Linear
        Fully connected layer
    :var cl      : torch.nn.Linear
        Final fully connected layer for classification
    """

    def __init__(self, num_classes):
        super(LDA_CIFAR, self).__init__()

        self.expected_input_size = (32, 32)

        # First layer
        self.conv1 = nn.Sequential(  # in: 32x32x3 out: 32x32x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1,
                      padding=1),
            nn.Softsign()
        )
        # Second layer
        self.conv2 = nn.Sequential(  # in: 32x32x16 out: 32x32x32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.Softsign()
        )
        # Third layer
        self.conv3 = nn.Sequential(  # in: 32x32x32 out: 32x32x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.Softsign()
        )
        # Fourth layer
        self.conv4 = nn.Sequential(  # in: 32x32x62 out: 16x16x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2,
                      stride=2),
            nn.Softsign()
        )
        # Fifth layer
        self.conv5 = nn.Sequential(  # in: 16x16x128 out: 8x8x192
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=2,
                      stride=2),
            nn.Softsign()
        )
        # Sixth layer
        self.conv6 = nn.Sequential(  # in: 8x8x192 out: 4x4x128
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=2,
                      stride=2),
            nn.Softsign()
        )

        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 128, out_features=num_classes),
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.cl(x)
        return x


class LDA_simple(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
        The convolutional layer of the network
    :var cl      : torch.nn.Linear
        Final fully connected layer for classification
    """

    def __init__(self):
        super(LDA_simple, self).__init__()

        self.expected_input_size = (32, 32)

        # First layer
        self.conv1 = nn.Sequential(  # in: 32x32x3 out: 32x32x16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4,
                      padding=0),
            nn.Softsign()
        )
        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 16, out_features=10),
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.cl(x)
        return x
