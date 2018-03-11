import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, output_channels=1000):
        super(AlexNet, self).__init__()

        self.expected_input_size = (227, 227)

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.Softsign(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.Softsign()
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Softsign(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Softsign()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Softsign(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d()
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Softsign(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Softsign(),
        )
        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(4096, output_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cl(x)
        return x

    def load_pretrained_state_dict(self, state_dict):
        own_state = self.state_dict()
        for own, pt in zip(own_state.keys(), state_dict.keys()):

            try:
                own_state[own].copy_(state_dict[pt].data)
            except:
                logging.debug('While copying the parameter named {}, whose dimensions in the model are'
                              ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    own, own_state[own].size(), state_dict[pt].size()))


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_pretrained_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
