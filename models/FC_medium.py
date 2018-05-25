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


class FC_medium(nn.Module):
    """
    Simple feed forward neural network

    Attributes
    ----------
    expected_input_size : int
        Expected input size
    fc1 : torch.nn.Sequential
    fc2 : torch.nn.Sequential
    fc3 : torch.nn.Sequential
        Fully connected layers of the network
    cl : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=10, **kwargs):
        """
        Creates an FC_medium model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
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

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.cl(x)
        return x
