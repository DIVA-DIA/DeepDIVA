"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn


class FC_simple(nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
        The first three convolutional layers of the network

    :var fc      : torch.nn.Linear
        Final fully connected layer
    """

    def __init__(self, num_classes=10, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(FC_simple, self).__init__()

        self.expected_input_size = (2)

        # First layer
        self.fc1 = nn.Sequential(
            nn.Linear(2, 10),
            nn.Softsign()
        )

        # Classification layer
        self.fc2 = nn.Sequential(
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x
