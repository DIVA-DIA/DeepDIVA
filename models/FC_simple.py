
import torch.nn as nn


class FC_simple(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(FC_simple, self).__init__()

        self.expected_input_size = (2)

        # First layer
        self.fc1 = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
        )

        # Classification layer
        self.fc2 = nn.Sequential(
            nn.Linear(4, num_classes)
        )

    def forward(self, x):
        """
        Computes forward pass on the network
        :param x: torch.Tensor
            The input to the model
        :return: torch.Tensor
            Activations of the fully connected layer
        """
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
