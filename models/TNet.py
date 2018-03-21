"""
Tfeat network from  http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
"""
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class TNet(nn.Module):
    """
    TFeat model definition
    """

    def __init__(self, output_channels=128, input_channels=3, **kwargs):
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
            nn.Tanh()
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
        x = self.fc(x)
        return x
