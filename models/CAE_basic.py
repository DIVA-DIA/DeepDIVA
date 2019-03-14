"""
Convolutional Auto Encoder with 3 conv layers and a fully connected classification layer
"""

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


class CAE_basic(nn.Module):
    """
    Simple convolutional auto-encoder neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    """

    def __init__(self, input_channels=3, output_channels=1,
                 auto_encoder_mode=False, return_features=False, heads_count=1, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        output_channels : int
            Number of neurons in the classification layers
        """
        super(CAE_basic, self).__init__()

        self.auto_encoder_mode = auto_encoder_mode
        self.heads_count = heads_count
        self.return_features = return_features
        self.expected_input_size = (96, 96)

        # Encoder layers ###############################################################################################
        # In: 96x96 Out: 32x32
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32,
                      kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        # In: 32x32 Out: 32x32
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        # In: 32x32 Out: 16x16
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96,
                      kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
        )
        # In: 16x16 Out: 8x8
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        # In: 8x8 Out: 4x4
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        # Decoder layers # Encoder layers ##############################################################################
        # In: 4x4 Out: 8x8
        self.dec_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        # In: 8x8 Out: 16x16
        self.dec_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=96,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
        )
        # In: 16x16 Out: 32x32
        self.dec_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=96, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        # In: 32x32 Out: 32x32
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # nn.Tanh(),
        )
        # In: 32x32 Out: 96x96
        self.dec_conv5 = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=3,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
        )
        # Hydra heads
        self.hydra = nn.ModuleList(
            [nn.Sequential(Flatten(), nn.Linear(4 * 4 * 32, output_channels)) for _ in range(self.heads_count)]
        )

    def encoder(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        x = self.enc_conv5(x)
        return x

    def decoder(self, x):
        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        x = self.dec_conv4(x)
        x = self.dec_conv5(x)
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
        x : Variable
            Activations of the heads or the reconstructed input
        features : Variable
            Features before the heads layers
        """
        x = self.encoder(x)

        # Store features
        features = Flatten()(x)

        if self.auto_encoder_mode:
            # Reconstruct the input
            x = self.decoder(x)
        else:
            # Compute output for all heads
            x = []
            for head in self.hydra:
                x.append(head(features))
            if self.heads_count == 1:
                x = x[0]

        if self.return_features:
            return x, features
        else:
            return x
