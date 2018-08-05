"""
This script generates visualizations of the activation of intermediate layers of CNNs.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

import models


def make_grid_own(activations):
    """
    Plots all activations of a layer in a grid format.
    Parameters
    ----------
    activations: numpy.ndarray
        array of activation values for each filter in a layer

    Returns
    -------
    large_fig: numpy.ndarray
        image array containing all activation heatmaps of a layer
    """
    activations = (activations / np.max(activations)) * 255
    num_plots = int(np.ceil(np.sqrt(activations.shape[0])))
    large_fig = np.zeros((num_plots * activations.shape[1], num_plots * activations.shape[2]))
    y_level = -1
    for idx in range(activations.shape[0]):
        if idx % num_plots == 0:
            y_level += 1
        beg_x = (idx % num_plots) * activations.shape[1]
        end_x = (idx % num_plots + 1) * activations.shape[1]
        beg_y = y_level * activations.shape[2]
        end_y = (y_level + 1) * activations.shape[2]
        large_fig[beg_x:end_x, beg_y:end_y] = activations[idx]
    return large_fig.astype(np.uint8)


def main(args):
    """
    Main routine of script to generate activation heatmaps.
    Parameters
    ----------
    args : argparse.Namespace
        contains all arguments parsed from input

    Returns
    -------
    None

    """
    model = models.__dict__[args.model_name](pretrained=args.pretrained)

    # Resume from checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            sys.exit(-1)

    img = Image.open(args.input_image)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img)
    img_tensor = torch.autograd.Variable(img_tensor.unsqueeze_(0))

    x = img_tensor

    for i, layer in enumerate(model.children()):
        x = layer(x)
        if i + 1 == args.layer:
            break

    img = x.data.permute(1, 0, 2, 3)

    img = make_grid(img, scale_each=True, normalize=True).numpy().transpose(1, 2, 0) * 255
    img = img.astype(np.uint8)

    img = Image.fromarray(img)
    img = img.resize(size=(1000, 1000), resample=Image.BICUBIC)
    img.save('/home/pondenka/output.png')

    # cv2.imwrite('/home/pondenka/output.png', img)

    print(x.data.numpy().shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        dest='model_name',
                        default='CNN_basic',
                        help='which model to use for training')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='path to latest checkpoint')
    parser.add_argument('--layer',
                        type=int,
                        default=None,
                        help='layer to visualize the activations from')
    parser.add_argument('--pretrained',
                        action='store_true',
                        default=False,
                        help='use pretrained model. (Not applicable for all models)')
    parser.add_argument('--input_image',
                        type=str,
                        default=None,
                        help='path to an input image')
    args = parser.parse_args()

    main(args)
