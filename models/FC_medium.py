import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class FC_medium(nn.Module):
    def __init__(self, output_channels=10, **kwargs):
        """
        :param output_channels: the number of classes in the dataset
        """
        super(FC_medium, self).__init__()

        self.expected_input_size = 2

        first_layer = 8
        second_layer = 8
        third_layer = 8

        # First layer
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(self.expected_input_size, first_layer),
            nn.Tanh(),
        )

        # Second layer
        self.fc2 = nn.Sequential(
            nn.Linear(first_layer, second_layer),
            nn.Tanh(),
        )

        # Third layer
        self.fc3 = nn.Sequential(
            nn.Linear(second_layer, third_layer),
            nn.Tanh(),
        )

        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(third_layer, output_channels)
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
        x = self.fc3(x)
        x = self.cl(x)
        return x
