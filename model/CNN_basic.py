"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch


class CNN_Basic(torch.nn.Module):
    """
    :var conv1   : torch.nn.Conv2d
    :var conv2   : torch.nn.Conv2d
    :var conv3   : torch.nn.Conv2d
        The first three convolutional layers of the network

    :var fc      : torch.nn.Linear
        Final fully connected layer

    :var af      : torch.nn.Softsign
        Activation function
    """

    def __init__(self, num_classes):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(CNN_Basic, self).__init__()
        # First layer
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=5, stride=3)
        # Second layer
        self.conv2 = torch.nn.Conv2d(24, 48, kernel_size=3, stride=2)
        # Third layer
        self.conv3 = torch.nn.Conv2d(48, 72, kernel_size=3, stride=2)
        # Classification layer
        self.fc = torch.nn.Linear(72, num_classes)
        # Activation function
        self.af = torch.nn.Softsign()

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        # Forward first layer
        c1 = self.conv1(x)
        c1_ss = self.af(c1)
        # Forward second layer
        c2 = self.conv2(c1_ss)
        c2_ss = self.af(c2)
        # Forward third layer
        c3 = self.conv3(c2_ss)
        c3_ss = self.af(c3)
        # Forward fully connected layer
        fc = self.fc(c3_ss.view(c3_ss.size()[0], -1))
        return fc
