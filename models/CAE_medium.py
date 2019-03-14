"""
Convolutional Auto Encoder with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class CAE_medium(nn.Module):
    """
    Simple convolutional auto-encoder neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, input_channels=3, return_features=False, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(CAE_medium, self).__init__()

        self.return_features = return_features
        self.expected_input_size = (32, 32)

        # Encoder layers
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(),
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(),
        )

        # Decoder layers
        # Third layer
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 128, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def decoder(self, x):
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

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
        x = self.encoder(x)

        if self.return_features:
            features = Flatten(x)
            return self.decoder(x), features
        else:
            x = self.decoder(x)
            return x
