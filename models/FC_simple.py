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


class FC_simple(nn.Module):
    """
    Simple feed forward neural network

    Attributes
    ----------
    expected_input_size : int
        Expected input size
    fc1 : torch.nn.Sequential
        Fully connected layer of the network
    cl : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=5, **kwargs):
        """
        Creates an FC_simple model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        """
        super(FC_simple, self).__init__()

        self.expected_input_size = 2

        hidden = 20

        # First layer
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(self.expected_input_size, hidden),
            nn.Tanh(),
        )

        # Classification layer
        self.fc2 = nn.Sequential(
            nn.Linear(hidden, output_channels)
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

        return x
