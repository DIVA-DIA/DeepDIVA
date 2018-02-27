import torch.nn as nn
import logging

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class FC_simple(nn.Module):
    def __init__(self, num_classes=5, leakyrelu=False, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(FC_simple, self).__init__()

        self.expected_input_size = 2

        if leakyrelu == False:
            logging.info('Using TanH for the model')
            # First layer
            self.fc1 = nn.Sequential(
                Flatten(),
                nn.Linear(2, 10),
                nn.Tanh(),
            )
        else:
            logging.info('Using LeakyReLu for the model')
            # First layer
            self.fc1 = nn.Sequential(
                Flatten(),
                nn.Linear(2, 10),
                nn.LeakyReLU(),
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
