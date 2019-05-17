"""
Convolutional Auto Encoder with 3 conv layers and a fully connected classification layer
"""

import torch.nn as nn
import torch
from models.registry import Model

@Model
def babyunet(output_channels=8, **kwargs):
    return BabyUnet(output_channels=output_channels)


class BabyUnet(nn.Module):
    """
    Simple convolutional auto-encoder neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    """
    def __init__(self, input_channels=3, output_channels=3, num_filter=24, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        output_channels : int
            Number of classes
        """

        super(BabyUnet, self).__init__()
        self.in_dim = input_channels
        self.out_dim = output_channels
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down = conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool = maxpool()

        self.bridge = conv_block(self.num_filter, self.num_filter * 2, act_fn)

        self.trans = conv_trans_block(self.num_filter * 2, self.num_filter, act_fn)
        self.up = conv_block(self.num_filter * 2, self.num_filter, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        down = self.down(input)
        pool = self.pool(down)

        bridge = self.bridge(pool)

        trans = self.trans(bridge)
        concat = torch.cat([trans, down], dim=1)
        up = self.up(concat)

        out = self.out(up)
        return out


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool