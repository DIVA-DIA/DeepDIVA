import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            nn.Softsign(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            nn.Softsign()
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Softsign(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Softsign()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Softsign(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Dropout2d()
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            nn.Softsign(),
        )
        self.fc2 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Softsign(),
        )
        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(4096, num_classes),
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
