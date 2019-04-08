import argparse
import os
from multiprocessing import Pool

import numpy as np

from util.misc import get_all_files_in_folders_and_subfolders, has_extension, load_numpy_image, save_numpy_image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def get_list_images(dir):
    images = get_all_files_in_folders_and_subfolders(dir)
    images = [item for item in images if has_extension(item, IMG_EXTENSIONS)]
    return images


def remove_empty(img):
    img = np.invert(img)
    x_locs = np.where(img.sum(axis=0).sum(axis=1) == 0)[0]
    y_locs = np.where(img.sum(axis=1).sum(axis=1) == 0)[0]
    img = np.delete(img, y_locs, axis=0)
    img = np.delete(img, x_locs, axis=1)
    img = np.invert(img)
    return img


def open_crop_save(path):
    img = load_numpy_image(path)
    img = remove_empty(img)
    os.remove(path)
    path = path.split('.')[0] + '.png'
    save_numpy_image(path, img)
    return


def main(args):
    pool = Pool(args.workers)
    images = get_list_images(args.root)
    pool.map(open_crop_save, images)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root',
                        type=str)
    parser.add_argument('-j',
                        '--workers',
                        type=int)
    args = parser.parse_args()
    main(args)
