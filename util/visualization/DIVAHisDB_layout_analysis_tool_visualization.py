import os
import numpy as np
import sys
import argparse

from PIL import Image

from util.misc import has_extension, pil_loader, make_colour_legend_image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# GREEN: Foreground predicted correctly rgb(70, 160, 30)
# YELLOW: Foreground predicted - but the wrong class (e.g. Text instead of Comment) rgb(255, 255, 60)
# BLACK: Background predicted correctly rgb(0, 0, 0)
# RED: Background mis-predicted as Foreground rgb(240, 30, 20)
# BLUE: Foreground mis-predicted as Background rgb(0, 240, 255)
CLASS_COLOUR_ENCODINGS = {"fg_correct": (70, 160, 30), "fg_wrong_class": (255, 255, 60), "bg_correct": (0, 0, 0),
             "bg_as_fg": (240, 30, 20), "fg_as_bg": (0, 240, 255)}


def layout_analysis_output(args):
    ground_truth_folder = args.ground_truth_folder
    network_output_folder = args.network_output_folder
    output_folder =  os.path.join(args.output_folder, 'layout_analysis_evaluation')

    gt_img_paths = get_img_paths(ground_truth_folder)
    segm_img_paths = get_img_paths(network_output_folder)

    if not output_folder:
        output_folder = os.path.join(os.path.dirname(segm_img_paths), 'layout_analysis_evaluation')

    for gt, segm in zip(gt_img_paths, segm_img_paths):
        gt_image = np.array(pil_loader(gt))
        segm_image = np.array(pil_loader(segm))

        generate_layout_analysis_output(output_folder, gt_image, segm_image, os.path.basename(gt))

    # create a legend and save it
    make_colour_legend_image(os.path.join(output_folder, "layout_analysis_eval_legend"), CLASS_COLOUR_ENCODINGS)


def generate_layout_analysis_output(output_folder, gt_image, segm_image, output_name, legend=False):
    """
    This function generates and saves an output like the one generated from the Output image as described in https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator

    Parameters
    ----------
    output_folder: string
        folder where the output is saved to
    gt_image: numpy array
        ground truth image in RGB
    segm_image: numpy array
        segmentation output in RGB
    output_name: string
        name of the output image
    legend: Boolaen
        set to true if legend should also be generated
    Returns
    -------
    None (saves output)
    """
    dest_filename = os.path.join(output_folder, output_name)
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img_la = np.zeros(segm_image.shape)

    # Extract just blue channel
    out_blue = segm_image[:, :, 2] #[2140:2145, 1570:1575]
    gt_blue = gt_image[:, :, 2] #[2140:2145, 1570:1575]

    # subtract the boundary pixel from the gt
    boundary_pixel = gt_image[:, :, 0].astype(np.uint8) == 128
    gt_blue[boundary_pixel] = 1

    # get the colour masks
    masks = {c: _get_mask(c, out_blue, gt_blue) for c in CLASS_COLOUR_ENCODINGS.keys()}

    # colour the pixels according to the masks
    for c, mask in masks.items():
        img_la[mask] = CLASS_COLOUR_ENCODINGS[c]

    # correct for boundary pixels
    boundary_masks = _get_boundary_masks(out_blue, gt_blue, boundary_pixel)
    # colour the pixels according to the masks
    for c, mask in boundary_masks.items():
        img_la[mask] = CLASS_COLOUR_ENCODINGS[c]

    # save the output
    result = Image.fromarray(img_la.astype(np.uint8))
    result.save(dest_filename)

    if legend:
        make_colour_legend_image(os.path.join(os.path.dirname(dest_filename), "layout_analysis_eval_legend"), CLASS_COLOUR_ENCODINGS)


def get_img_paths(path_imgs):
    if not os.path.isdir(path_imgs):
        print("Folder data or gt not found in " + path_imgs)

    images = []

    for _, _, fnames in sorted(os.walk(path_imgs)):
        for fname in sorted(fnames):
            if has_extension(fname, IMG_EXTENSIONS):
                path_img = os.path.join(path_imgs, fname)
                images.append(path_img)

    return images


def _get_mask(tag, output, gt):
    if tag == "fg_correct":
        return np.logical_and(output == gt, gt != 1)
    elif tag == "bg_correct":
        return np.logical_and(output == gt, gt == 1)
    elif tag == "fg_wrong_class":
        return np.logical_and(output != gt, np.logical_and(gt != 1, output != 1))
    elif tag == "fg_as_bg":
        return np.logical_and(output != gt, output == 1)
    elif tag == "bg_as_fg":
        return np.logical_and(output != gt, np.logical_and(output != 1, gt == 1))


def _get_boundary_masks(output, gt, boundary_pixel):
    masks = {}

    # background pixels
    masks["bg_correct"] = np.logical_and.reduce([boundary_pixel, output != gt, output != 1, gt == 1])
    # foreground pixels
    masks["fg_correct"] = np.logical_and.reduce([boundary_pixel, output != gt, output == 1, gt != 1])

    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to generate a visualization '
                                                 'similar to the DIVAHisDB Layout Analysis Tool')

    parser.add_argument('--ground-truth-folder',
                        help='path to the ground truth images',
                        type=str)
    parser.add_argument('--network-output-folder',
                        help='path to the segmentation output of the network',
                        type=str)
    parser.add_argument('--output-folder',
                        help='path to where the visualizations should be saved to. If none is provided it is saved '
                             'in the same folder as the output of the network',
                        required=False,
                        type=str,
                        default=None)

    args = parser.parse_args()

    layout_analysis_output(args)
