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

    def __init__(self, **kwargs):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(64 * 8 * 8, 128),
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
        x = self.features(x)
        x = self.classifier(x)
        return x
