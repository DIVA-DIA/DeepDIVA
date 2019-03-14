import math
import torch.nn as nn

__all__ = ['BabyResNet', 'babyresnet18', 'babyresnet34', 'babyresnet50', 'babyresnet101', 'babyresnet152']

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class BabyResNet(nn.Module):
    r"""
    ResNet model architecture adapted from `<https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>`
    It is better suited for smaller images as the expected input size is TODO

    Attributes
    ----------
    features : torch.Tensor
        Features just before the final fully connected layer
    inplanes : int
        Number of output dimensions of the first conv1 layer
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
        First conv2d+bn+relu+maxpool operation to bring the input at size 56x56x64,
        which is the expected input size of conv2x
    conv2x : torch.nn.Sequential
    conv3x : torch.nn.Sequential
    conv4x : torch.nn.Sequential
    conv5x : torch.nn.Sequential
        groups of "layers" for the network. Each of them is composed of multiple sub-block
        (either _BasickBlock or _BottleNeck) depending on the size of the network (ResNet 18,34,50,101,152)
    avgpool :
        The final average pool before the fully connected layer
    fc : torch.nn.Linear
       Fully connected layer for classification
    """

    def __init__(self, block_type, num_block, output_channels=1000, **kwargs):
        """
        Creates a BabyResNet model from the scratch.

        BabyResNet differs from a regular ResNet for it has an input size of 32x32
        rather than 224x224. The intuition behind is that it makes not so much sense
        to rescale an input image from a native domain from 32x32 to 224x224 (e.g. in
        CIFAR or MNIST datasets) because of multiple reasons. To begin with,  the image
        distortion of such an upscaling is massive, but most importantly we pay an
        extremely high overhead both in terms of computation and in number of parameters
        involved. For this reason the ResNet architecture has been adapted to fit a
        smaller input size.

        Adaptations:
        The conv1 layer filters have been reduced from 7x7 to 3x3 with stride 1 and the
        padding removed. Additionally the number of filters has been increased to 128.
        The initial maxpool stride has been lowered to 1.
        This way, with an input of 32x32 the final output of the conv1 is then 28x28x128
        which matches the expected input size of conv3x layer. This is no coincidence.
        Since the image is already smaller than 56x56 (which is the expected size of
        conv2x layer) the conv2x layer has been dropped entirely.
        This would reduce the total number of layers in the network. In an effort to
        reduce this gap, we increased the number of blocks in the conv3x layer with
        as many blocks there where in the conv2x (we basically moved blocks from one layer
        to another).
        The final architecture closely matches the original ResNet but it is optimized
        for handling an input of 32x32 pixels. The results of the BabyResNet and the original
        ResNet are NOT the same (the final number of parameters differs slightly with the
        BabyResNet having 320 parameters more) and the BabyResNet obtain (as expected)
        better results on CIFAR-10.

        Parameters
        ----------
        block_type : nn.Module
            Type of the blocks to be used in the network. Either _BasickBlock or _BottleNeck.
        num_block : List(int)
            Number of blocks to put in each layer of the network. Must be of size 4
        output_channels : int
            Number of neurons in the last layer
        """
        super(BabyResNet, self).__init__()

        self.features = None
        self.num_input_filters = 128  # Attention: this gets updated after each convx layer creation!
        self.expected_input_size = (32, 32)

        # First convolutional layer, bring the input into the 56x56x64 desired size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.num_input_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.num_input_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
        )

        # Bulk part of the ResNet with four groups of blocks (expected size: 56x56x64)
        self.conv3x = self._make_layer(block_type, 128, num_block[1])
        self.conv4x = self._make_layer(block_type, 256, num_block[2], stride=2)
        self.conv5x = self._make_layer(block_type, 512, num_block[3], stride=2)

        # Final averaging and fully connected layer for classification (expected size: 7x7x512*block_type.expansion)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512 * block_type.expansion, output_channels),
        )

        # Initialize the weights of all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize with "Xavier". Biases are not initialized because they are not used in conv.
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block_type, filters, num_block, stride=1):
        """
        This function creates a convx layer for the ResNet.
        A convx layer is a group of blocks (either _BasickBlock or _BottleNeck)
        where each block is skipped by a connection. Refer to the original paper for more info.

        Parameters
        ----------
        block_type : nn.Module
            Type of the blocks to be used in the network. Either _BasickBlock or _BottleNeck.
        filters : int
            Number of filters to be used in the two convolutional layers inside a block
        num_block : List(int)
            Number of blocks to put in each layer of the network. Must be of size 4
        stride : int
            Specifies the stride. It is used also to flag when the residual dimensions have to be halved
        Returns
        -------
        torch.nn.Sequential
            The convx layer as a sequential containing all the blocks
        """
        downsample = None
        if stride != 1 or self.num_input_filters != filters * block_type.expansion:
            # This modules halves the dimension of the residual connection (dotted line Fig.3 of the paper)
            # Also beware that in the case of _Bottleneck block this could up upsampling the residual to an higher dimension!
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.num_input_filters, out_channels=filters * block_type.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters * block_type.expansion),
            )

        layers = []
        layers.append(block_type(self.num_input_filters, filters, stride, downsample))
        self.num_input_filters = filters * block_type.expansion
        for i in range(1, num_block):
            layers.append(block_type(self.num_input_filters, filters))

        return nn.Sequential(*layers)

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
        x = self.conv1(x)

        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)

        x = self.avgpool(x)
        self.features = x
        x = self.fc(x)
        return x


class _BasicBlock(nn.Module):
    """
    This is the basic block of a ResNet.
    """
    expansion = 1

    def __init__(self, in_filters, out_filters, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_filters, out_channels=out_filters,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.downsample = downsample

    def forward(self, x):
        """
        Computes forward pass on the block

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on

        Returns
        -------
        Variable
            Activations of the block summed to the residual connection values
        """
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class _Bottleneck(nn.Module):
    """
    This is the bottleneck block of a ResNet. It is used in the deeper version for performance reason.
    """
    expansion = 4

    def __init__(self, in_filters, out_filters, stride=1, downsample=None):
        super(_Bottleneck, self).__init__()
        # 1x1 conv to bring it down to 'filters (64,128,256,512)'
        self.conv1 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        # 3x3 conv full size
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        # 1x1 conv to bring in up to the input size
        self.conv3 = nn.Conv2d(out_filters, out_filters * _Bottleneck.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_filters * _Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        Computes forward pass on the block

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on

        Returns
        -------
        Variable
            Activations of the block summed to the residual connection values
        """
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


def babyresnet18(**kwargs):
    """
    Constructs a _BabyResNet-18 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return BabyResNet(_BasicBlock, [0, 4, 2, 2], **kwargs)


def babyresnet34(**kwargs):
    """
    Constructs a _BabyResNet-34 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return BabyResNet(_BasicBlock, [0, 7, 6, 3], **kwargs)


def babyresnet50(**kwargs):
    """
    Constructs a _BabyResNet-50 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return BabyResNet(_Bottleneck, [0, 7, 6, 3], **kwargs)


def babyresnet101(**kwargs):
    """
    Constructs a _BabyResNet-101 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return BabyResNet(_Bottleneck, [0, 7, 23, 3], **kwargs)


def babyresnet152(**kwargs):
    """
    Constructs a _BabyResNet-152 model.

    Returns
    -------
    torch.nn.Module
        The model of the network
    """
    return BabyResNet(_Bottleneck, [0, 11, 36, 3], **kwargs)
