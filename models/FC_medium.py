import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class FC_medium(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        """
        :param num_classes: the number of classes in the dataset
        """
        super(FC_medium, self).__init__()

        self.expected_input_size = 2

        # First layer
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(2, 40),
            nn.LeakyReLU(),
        )

        # Second layer
        self.fc2 = nn.Sequential(
            nn.Linear(40, 60),
            nn.LeakyReLU(),
        )

        # Third layer
        self.fc3 = nn.Sequential(
            nn.Linear(60, 80),
            nn.LeakyReLU(),
        )

        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(80, num_classes)
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
