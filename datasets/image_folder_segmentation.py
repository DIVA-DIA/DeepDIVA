"""
Load a dataset of historic documents by specifying the folder where its located.
"""

# Utils
import logging
import os
import random
import sys
import math
import os.path
from collections import deque
import numpy as np

# from DeepDIVA
from template.setup import _load_class_encodings
from util.misc import has_extension, pil_loader
from .custom_transform_library.transforms import ToTensorTwinImage, ToTensorSlidingWindowCrop

# Torch related stuff
import torch.utils.data as data

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def get_gt_data_paths(directory):
    """
    Parameters
    ----------
    directory: string
        parent directory with gt and data folder inside

    Returns
    -------
    paths: list of tuples

    """

    paths = []
    directory = os.path.expanduser(directory)

    path_imgs = os.path.join(directory, "data")
    path_gts = os.path.join(directory, "gt")

    if not (os.path.isdir(path_imgs) or os.path.isdir(path_gts)):
        logging.error("folder data or gt not found in " + str(directory))

    for img_name, gt_name in zip(sorted(os.listdir(path_imgs)), sorted(os.listdir(path_gts))):
        if has_extension(img_name, IMG_EXTENSIONS) and has_extension(gt_name, IMG_EXTENSIONS):
            paths.append((os.path.join(path_imgs, img_name), os.path.join(path_gts, gt_name)))

    return paths


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'PIL':
        return pil_loader(path)
    else:
        logging.info("Something went wrong with the default_loader in image_folder_segmentation.py")
        sys.exit(-1)


def load_dataset(dataset_folder, workers, in_memory=False, **kwargs):
    """
    Loads the dataset from file system and provides the dataset splits for train validation and test

    The dataset is expected to be in the following structure, where 'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/DIVAHisDB/CB55/"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    In each of the three splits (train, val, test) there are two folders. One for the ground truth ("gt")
    and the other for the images ("data"). The ground truth image is of equal size and and encoded the classes
    are encoded in the blue channel. The images that belong together in data and gt need to have the same name (but
    the can have a different image extension).

    Example:

        ../CB55/train/data/page23.png
        ../CB55/train/data/page231.png
        ../CB55/train/gt/page23.png
        ../CB55/train/gt/page231.png

        ../CB55/val/data
        ../CB55/val/gt
        ../CB55/test/data
        ../CB55/test/gt



    Parameters
    ----------
    dataset_folder : string
        Path to the dataset on the file System

    args : dict
        Dictionary of all the CLI arguments passed in

    in_memory : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.

    workers: int
        Number of workers to use for the dataloaders

    testing: boolean
        Take another path if you are in testing phase

    Returns
    -------
    train_ds : data.Dataset

    val_ds : data.Dataset

    test_ds : data.Dataset
        Train, validation and test splits
    """
    # Sanity check on the splits folders
    if not os.path.isdir(dataset_folder):
        logging.error("Dataset folder not found at " + dataset_folder)
        sys.exit(-1)

    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    if in_memory:
        logging.error("With segmentation you don't have the option to put everything into memory")
        sys.exit(-1)

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # get the class encodings
    classes = _load_class_encodings(dataset_folder, inmem=in_memory, workers=workers, **kwargs)

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir, classes, workers, **kwargs)
    val_ds = ImageFolder(val_dir, classes, workers, **kwargs)
    # the number of workers has to be 1 during testing (concurrency issues)
    test_ds = ImageFolder(test_dir, classes, num_workers=1, is_test=True, **kwargs)
    return train_ds, val_ds, test_ds


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png
    """

    def __init__(self, root, class_encodings, num_workers, imgs_in_memory, crops_per_image, crop_size,
                 transform=None, img_transform=None, gt_transform=None, loader=default_loader,
                 is_test=False, **kwargs):
        """
        #TODO doc
        Parameters
        ----------
        root : string
            Path to dataset folder (train / val / test)
        class_encodings :
        num_workers : int
        imgs_in_memory :
        crops_per_image : int
        crop_size : int
        transform : callable
        img_transform : callable
        gt_transform : callable # TODO why are there 3 transforms?
        loader : callable
            A function to load an image given its path.
        """

        # Init list
        self.root = root
        self.class_encodings = class_encodings
        self.num_classes = len(self.class_encodings)
        self.num_workers = num_workers
        self.imgs_in_memory = imgs_in_memory
        self.crops_per_image = crops_per_image if not is_test else None
        self.crop_size = crop_size
        self.transform = transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.loader = loader
        self.is_test = is_test

        # List of tuples that contain the path to the gt and image that belong together
        self.img_paths = get_gt_data_paths(root)
        self.num_imgs_in_set = len(self.img_paths)
        if self.num_imgs_in_set == 0:
            raise RuntimeError("Found 0 images in subfolders of: {} \n Supported image extensions are: {}".format(
                root, ",".join(IMG_EXTENSIONS)))

        # Make sure work can be split into workers equally
        if self.num_workers > 1:
            # Length is divisible by the number of workers
            if self.__len__() % self.num_workers != 0:
                logging.error("{} (number of pages in set ({}) * crops per image {}) "
                              "must be divisible by the number of workers (currently {})".format(
                    self.__len__(), self.num_imgs_in_set, self.crops_per_image, self.num_workers))
                sys.exit(-1)
            # Crops per page is divisible by the number of workers
            if self.crops_per_image % self.num_workers != 0:
                logging.error("{} (# crops per page) must be divisible by the number of"
                              " workers (currently {})".format(self.crops_per_image, self.num_workers))
                sys.exit(-1)

        # Index in self.image_order of the next image to be loaded
        self.next_image_index = 0

        # Keeping track of how many crops have been generated -> needed to know when to shuffle the pages
        self.current_number_of_crops = 0  # set to zero again when new page / page-bundle is loaded

        if self.is_test:
            # Overlap for the sliding window (% of crop size)
            self.overlap = 0.5
            # Get the numbers for __len__
            self.img_names_sizes, self.num_horiz_crops, self.num_vert_crops = self._get_img_size_and_crop_numbers()
        else:
            self.crops_per_image_per_worker = crops_per_image // self.num_workers
            # List with the index order of the images
            self.image_order = list(range(self.num_imgs_in_set))

        # Only used for debugging
        self.updated = 0
        self.img_and_updates = {os.path.basename(name[0]): 0 for name in self.img_paths}
        self.img_and_num_crops = {os.path.basename(name[0]): 0 for name in self.img_paths}

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop.
        The length is different during train/val and test, because we process the whole image during testing,
        and only sample from the images during train/val.
        """
        if self.is_test:
            # Sliding window
            return sum([hc*vc for hc, vc in zip(self.num_horiz_crops, self.num_vert_crops)])
        else:
            # Number of images in the dataset * how many crops per page
            return len(self.img_paths) * self.crops_per_image

    def __getitem__(self, index):
        # TODO documentation format
        """
        Args:
            index (int): Index

        Returns:
            during train and val
            tuple: (img_input, gt_target)

            during test
            tuple: ((img_input, orig_img_shape, top_left_coordinates_of_crop, test_img_name), gt_target)
                default sliding window (used during test)
        """
        if self.is_test:
            # Testing: sliding window with overlap
            if self.current_number_of_crops == 0:
                # load page
                self._load_test_images_and_vars()
            return self._get_test_items()
        else:
            return self._get_train_val_items()

    def _get_train_val_items(self):
        # TODO documentation
        # New bundle?
        if self.current_number_of_crops % self._bundle_len() == 0:
            self.current_number_of_crops = 0
            # Load new page
            self._load_page()
            # Shuffle the image bundle order at the beginning and when a new page is loaded
            random.shuffle(self.image_bundle_order)

        logging.debug("PID{}: Image order [{}]: {}".format(os.getpid(), len(self.image_order), self.image_order))
        logging.debug("PID{}: Image bundle order [{}]: {}".format(os.getpid(), self._bundle_len(), self.image_bundle_order))
        logging.debug("PID{}: Current number of crops: {}".format(os.getpid(), self.current_number_of_crops))

        current_img = self.imgnames_inmem[self.image_bundle_order[self.current_number_of_crops]]
        logging.debug("PID{}: Cropping from image: {}".format(os.getpid(), current_img))

        self.img_and_num_crops[current_img] = self.img_and_num_crops[current_img] + 1
        logging.debug("**********{}: crops/img {}".format(os.getpid(), self.img_and_num_crops))

        # get the items
        img, gt = self.apply_transformation(self.data_img_inmem[self.image_bundle_order[self.current_number_of_crops]],
                                            self.gt_img_inmem[self.image_bundle_order[self.current_number_of_crops]])

        # Update total number of crops
        self.current_number_of_crops += 1

        return img, gt

    def _get_test_items(self):
        # TODO documentation
        # get and update the coordinates for the sliding window
        coordinates = self._get_crop_coordinates()
        x_position, y_position = coordinates

        img, gt = self.apply_transformation(self.current_data_img, self.current_gt_img, coordinates=coordinates)

        # update total number of crops -> set to zero when last crop of epoch is generated
        self.current_number_of_crops = (self.current_number_of_crops + 1) % self.tot_crops_current_img
        self._update_sliding_window_coordinates()

        logging.debug("PID{}: Cropping position ({},{}). Horizontal {}/{}. Vertical {}/{}. Total {}/{}".format(
            os.getpid(), x_position, y_position, self.current_horiz_crop, self.current_num_horiz_crops,
            self.current_vert_crop, self.current_num_vert_crops, self.current_number_of_crops,
            self.tot_crops_current_img))

        return (img, coordinates, self.img_names_sizes[self.next_image_index-1][0]), gt

    def _load_test_images_and_vars(self):
        # TODO documentation
        # load image
        self.current_data_img = pil_loader(self.img_paths[self.next_image_index][0])
        self.current_gt_img = pil_loader(self.img_paths[self.next_image_index][1])

        # initialize the sliding window indices
        self.current_num_horiz_crops = self.num_horiz_crops[self.next_image_index]
        self.current_horiz_crop = 0
        self.current_num_vert_crops = self.num_vert_crops[self.next_image_index]
        self.current_vert_crop = 0

        self.tot_crops_current_img = self.current_num_vert_crops * self.current_num_horiz_crops

        # update pointer to next image
        self.next_image_index += 1

    def apply_transformation(self, img, gt, coordinates=None):
        """
        Applies the transformations that have been defined in the setup (setup.py). If no transformations
        have been defined, the PIL image is returned instead.

        Parameters
        ----------
        img: PIL image
            image data
        gt: PIL image
            ground truth image
        coordinates: tuple (int, int)
            coordinates where the sliding window should be cropped
        Returns
        -------
        tuple
            img and gt after transformations
        """
        if self.transform is not None:
            # perform transformations
            img, gt = self.transform(img, gt)

        # convert to tensor
        if self.is_test:
            # crop for sliding window
            img, gt = ToTensorSlidingWindowCrop(self.crop_size)(img, gt, coordinates)
        else:
            img, gt = ToTensorTwinImage()(img, gt)

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt

    def _bundle_len(self):
        return self.crops_per_image_per_worker * np.min([len(self.img_paths), self.imgs_in_memory])

    def _worker_len(self):
        return int(self.crops_per_image_per_worker * self.num_imgs_in_set)

    def _load_page(self):
        # TODO documentation
        # Get path to images to load. IMPORTANT: notice that the 2* below makes the page list a ring in this scenario
        # i.e. at the end we re-sample from the first pages if necessary.
        #start_index = np.min([self.next_image_index, len(self.image_order)- self.imgs_in_memory])  # Ensures that in memory there are always self.imgs_in_memory many images
        to_load = [self.img_paths[i] for i in (2*self.image_order)[self.next_image_index:(self.imgs_in_memory + self.next_image_index)]]

        self.imgnames_inmem = [os.path.basename(data_path) for (data_path, _) in to_load]
        self.data_img_inmem = [self.loader(data_path) for (data_path, _) in to_load]
        self.gt_img_inmem = [self.loader(gt_path) for (_, gt_path) in to_load]

        # update index of next image
        self.next_image_index += self.imgs_in_memory
        logging.debug("PID{}: Current images: {}".format(os.getpid(),
                                                         [os.path.basename(self.img_paths[i][0]) for i in self.image_order[(self.next_image_index-self.imgs_in_memory):self.next_image_index]]))
        # set back to zero if we get outside of the image range
        if self.next_image_index >= self.num_imgs_in_set:
            self.next_image_index = 0

        # gt and data images are the same size
        for img, gt in zip(self.data_img_inmem, self.gt_img_inmem):
            assert img.size == gt.size

        # create the order in which the crops are sample from the loaded images
        self.image_bundle_order = list(range(len(self.data_img_inmem))) * self.crops_per_image_per_worker

        # TODO remove
        # if len(self.image_bundle_order) != self._bundle_len():
        #     logging.debug("{} = {} * {}".format(len(self.image_bundle_order), len(self.data_img_inmem), self.crops_per_image_per_worker))
        #     logging.debug("{} = {} * {}".format(self._bundle_len(), self.crops_per_image_per_worker,  np.min([len(self.img_paths), self.imgs_in_memory])))
        #     sys.exit(-1)

        for (data_path, _) in to_load:
            self.img_and_updates[os.path.basename(data_path)] = self.img_and_updates[os.path.basename(data_path)] + 1
        logging.debug("**********{}: page updates {}".format(os.getpid(), self.img_and_updates))

    def _get_img_size_and_crop_numbers(self):
        # TODO documentation
        img_names_sizes = [] # list of tuples -> (gt_img_name, img_size (H, W))
        num_horiz_crops = []
        num_vert_crops = []

        for img_path, gt_path in self.img_paths:
            data_img = self.loader(img_path)
            gt_img = self.loader(gt_path)
            # Ensure that data and gt image are of the same size
            assert gt_img.size == data_img.size
            img_names_sizes.append((os.path.basename(gt_path), data_img.size[::-1]))
            num_horiz_crops.append(math.ceil(data_img.size[0] / (self.crop_size / (1 / self.overlap))))
            num_vert_crops.append(math.ceil(data_img.size[1] / (self.crop_size / (1 / self.overlap))))

        return img_names_sizes, num_horiz_crops, num_vert_crops

    def _update_sliding_window_coordinates(self):
        # TODO documentation
        self.current_horiz_crop = (self.current_horiz_crop + 1) % self.current_num_horiz_crops

        if self.current_horiz_crop == 0 or self.current_vert_crop == 0:
            self.current_vert_crop = (self.current_vert_crop + 1)

    def _get_crop_coordinates(self):
        # TODO documentation
        # x coordinate
        if self.current_horiz_crop == (self.current_num_horiz_crops - 1):
            # we are at the end of a line
            x_position = self.img_names_sizes[self.next_image_index-1][1][0] - self.crop_size
        else:
            # move one position to the right
            x_position = int(self.crop_size / (1 / self.overlap)) * self.current_horiz_crop

        # y coordinate
        if self.current_vert_crop == self.current_num_vert_crops:
            # we are at the bottom end
            y_position = self.img_names_sizes[self.next_image_index-1][1][1] - self.crop_size
        else:
            y_position = int(self.crop_size / (1 / self.overlap)) * self.current_vert_crop

        return x_position, y_position
