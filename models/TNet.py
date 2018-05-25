
import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class TNet(nn.Module):
    r"""
    Tfeat network from http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf

    Attributes
    ----------
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=128, input_channels=3, **kwargs):
        """
        Creates an TNet model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(TNet, self).__init__()

        self.expected_input_size = (32, 32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(64 * 8 * 8, output_channels),
        )

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x
