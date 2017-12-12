"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn


class CNN_Basic(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
        The first three convolutional layers of the network

    :var fc      : torch.nn.Linear
        Final fully connected layer
    """

    def __init__(self, num_classes, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(CNN_Basic, self).__init__()

        self.expected_input_size = (32, 32)

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=3),
            nn.Softsign()
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2),
            nn.Softsign()
        )
        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, stride=1),
            nn.Softsign()
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(288, num_classes)
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
