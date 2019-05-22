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
from collections import deque, OrderedDict

# from DeepDIVA
from template.setup import class_encodings
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
    classes = class_encodings(dataset_folder, inmem=in_memory, workers=workers, **kwargs)

    # Get an online dataset for each split
    train_ds = ImageFolder(train_dir, classes, workers, **kwargs)
    val_ds = ImageFolder(val_dir, classes, workers, **kwargs)
    # the number of workers has to be 1 during testing (concurrency issues)
    test_ds = ImageFolder(test_dir, classes, workers=1, **kwargs)
    return train_ds, val_ds, test_ds


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/gt/xxx.png
        root/gt/xxy.png
        root/gt/xxz.png

        root/data/xxx.png
        root/data/xxy.png
        root/data/xxz.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, classes, workers, imgs_in_memory, crops_per_image, crop_size,
                 transform=None, img_transform=None, gt_transform=None,
                 loader=default_loader, **kwargs):

        img_paths = get_gt_data_paths(root)
        # the total number of crops needs to be divisible by the number of workers
        self.num_workers = workers

        self.updated = 0
        self.img_and_updates = {os.path.basename(name[0]):0 for name in img_paths}
        self.img_and_num_cropos = {os.path.basename(name[0]):0 for name in img_paths}

        if len(img_paths) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\nSupported image extensions are: "
                                + ",".join(IMG_EXTENSIONS)))

        # path to dataset folder (train / val / test)
        self.root = root
        # list of tuples that contain the path to the gt and image that belong together
        self.img_paths = img_paths
        self.total_number_of_images = len(self.img_paths)

        # true if it is the test set
        self.is_test = "test" == os.path.basename(self.root)

        # transformations and loader
        self.transform = transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        # class encoding (list)
        self.class_encodings = classes
        self.num_classes = len(self.class_encodings)

        # set up iterators and parameters
        self.imgs_in_memory = imgs_in_memory
        self.crop_size = crop_size

        # index in self.image_order of the next image to be loaded
        self.next_image_index = 0

        # keeping track of how many crops have been generated -> needed to know when to shuffle the pages
        self.number_of_crops = 0  # total number per worker (__len__ / num_workers)
        self.current_number_of_crops = 0  # set to zero again when new page / page-bundle is loaded

        if self.is_test:
            # overlap for the sliding window (% of crop size)
            self.overlap = 0.5
            # get the numbers for __len__
            self.img_names_sizes, self.num_horiz_crops, self.num_vert_crops = self._get_img_size_and_crop_numbers()

        else:
            self.crops_per_image = crops_per_image
            # list with the index order of the images
            self.image_order = [i for i in range(self.total_number_of_images)]
            self.image_bundle_order = [i for i in range(self.imgs_in_memory)] * (self._bundle_len() // self.imgs_in_memory)

        # make sure length is divisible by the number of workers
        if self.num_workers > 1:
            if not self.__len__() % self.num_workers == 0:
                logging.error("{} (number of pages in set ({}) * images in memory * crops per image) must be divisible by the number of workers (currently {})".format(self.__len__(), len(self.img_paths), self.num_workers))
                sys.exit(-1)
            if not self.imgs_in_memory * self.crops_per_image % self.num_workers == 0:
                logging.error("{} (images in memory * crops per image) must be divisible by the number of workers (currently {})".format(self.imgs_in_memory * self.crops_per_image, self.num_workers))
                sys.exit(-1)

    def __len__(self):
        """
        This function returns the length of an epoch so the data loader knows when to stop. The length is different during
        train / val or test, because we process the whole image during testing, and only sample from the images during
        train / val.

        """
        if self.is_test:
            # sliding window
            return sum([hc*vc for hc, vc in zip(self.num_horiz_crops, self.num_vert_crops)])
        else:
            # number of images returned for random cropping
            return len(self.img_paths) * self.imgs_in_memory * self.crops_per_image

    def __getitem__(self, index):
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
        if not self.is_test:
            # training / validation
            # shuffle page order at start of epoch
            if self.number_of_crops == 0:
                self._shuffle_img_order()
                # initiate the queues where the images in memory are stored
                self.data_img_queue, self.gt_img_queue = self._initiate_queues()

            return self._get_train_val_items()

        else:
            # testing: sliding window with overlap
            if self.current_number_of_crops == 0:
                # load page
                self._load_test_images_and_vars()

            return self._get_test_items()

    def _get_train_val_items(self):
        if self.current_number_of_crops == 0:
            # shuffle the image bundle order at the beginning and when a new page is loaded
            self._shuffle_img_order_bundle()
            # check if new page needs to be loaded
            if self.number_of_crops > 0 and self.number_of_crops % self._bundle_len() == 0:
                self._update_queues()

        # logging.info("PID{}: Image order: {}".format(os.getpid(), self.image_order))
        # logging.info("PID{}: Image bundle order: {}".format(os.getpid(), self.image_bundle_order))
        # logging.info("PID{}: Current images: {}".format(os.getpid(),
        #     [os.path.basename(self.img_paths[i][0]) for i in self.image_order[(self.next_image_index-self.imgs_in_memory):self.next_image_index]]))
        # logging.info("PID{}: Cropping from image: {}".format(os.getpid(), os.path.basename(self.img_paths[self.image_order[
        #     (self.next_image_index - 1) - (self.imgs_in_memory - 1 - self.image_bundle_order[self.current_number_of_crops])
        #     ]][0])))
        current_img = os.path.basename(self.img_paths[self.image_order[
                                     (self.next_image_index - 1) - (self.imgs_in_memory - 1 - self.image_bundle_order
                                     [self.current_number_of_crops])]][0])
        self.img_and_num_cropos[current_img] = self.img_and_num_cropos[current_img] + 1
        # logging.info("**********{}: crops/img {}".format(os.getpid(), self.img_and_num_cropos))

        # get the items
        img, gt = self.apply_transformation(self.data_img_queue[self.image_bundle_order[self.current_number_of_crops]],
                                            self.gt_img_queue[self.image_bundle_order[self.current_number_of_crops]])

        # logging.info("PID{}: Crop number {} / {} of the bundle / {} total".format(os.getpid(),
        #              self.current_number_of_crops+1, self._bundle_len(), self._worker_len()))

        # update total number of crops
        # set to zero when last crop of image bundle is generated
        self.current_number_of_crops = (self.current_number_of_crops + 1) % (self._bundle_len())
        # set to zero when last crop of epoch is generated
        self.number_of_crops = (self.number_of_crops + 1) % (self._worker_len())

        return img, gt

    def _get_test_items(self):
        # get and update the coordinates for the sliding window
        coordinates = self._get_crop_coordinates()
        x_position, y_position = coordinates

        img, gt = self.apply_transformation(self.current_data_img, self.current_gt_img, coordinates=coordinates)

        # update total number of crops -> set to zero when last crop of epoch is generated
        self.current_number_of_crops = (self.current_number_of_crops + 1) % self.tot_crops_current_img
        self._update_sliding_window_coordinates()

        # logging.info("PID{}: Cropping position ({},{}). Horizontal {}/{}. Vertical {}/{}. Total {}/{}".format(
        #     os.getpid(), x_position, y_position, self.current_horiz_crop, self.current_num_horiz_crops,
        #     self.current_vert_crop, self.current_num_vert_crops, self.current_number_of_crops,
        #     self.tot_crops_current_img))

        return (img, coordinates, self.img_names_sizes[self.next_image_index-1][0]), gt

    def _load_test_images_and_vars(self):
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
        return self._worker_len() // len(self.img_paths)

    def _worker_len(self):
        return self.__len__() // self.num_workers

    def _shuffle_img_order(self):
        random.shuffle(self.image_order)

    def _shuffle_img_order_bundle(self):
        random.shuffle(self.image_bundle_order)

    def _initiate_queues(self):
        # first time loading images
        to_load = [self.img_paths[i] for i in self.image_order[:self.imgs_in_memory]]
        data_img_queue = deque([pil_loader(data_path) for (data_path, _) in to_load])
        gt_img_queue = deque([pil_loader(gt_path) for (_, gt_path) in to_load])

        # update index of next image
        self.next_image_index = self.imgs_in_memory

        # gt and data images are the same size
        for img, gt in zip(data_img_queue, gt_img_queue):
            assert img.size == gt.size

        for (data_path, _) in to_load:
            self.img_and_updates[os.path.basename(data_path)] = self.img_and_updates[os.path.basename(data_path)] + 1
            #print(os.getpid(), os.path.basename(data_path))
        #logging.info("**********{}: page udpates {}".format(os.getpid(), self.img_and_updates))

        return data_img_queue, gt_img_queue

    def _update_queues(self):
        # updating image queues
        # remove last element
        self.data_img_queue.pop()
        self.gt_img_queue.pop()

        # load new image to the front of the queue
        data_path, gt_path = self.img_paths[self.image_order[self.next_image_index]]
        self.data_img_queue.appendleft(pil_loader(data_path))
        self.gt_img_queue.appendleft(pil_loader(gt_path))

        # gt and data images are the same size
        assert self.data_img_queue[0].size == self.gt_img_queue[0].size

        self.img_and_updates[os.path.basename(data_path)] = self.img_and_updates[os.path.basename(data_path)] + 1
        #logging.info("**********{}: page udpates {}".format(os.getpid(), self.img_and_updates))
        # print(os.getpid(), os.path.basename(data_path))

        # update index of next image
        self.next_image_index = (self.next_image_index + 1) % self.total_number_of_images

    def _get_img_size_and_crop_numbers(self):
        img_names_sizes = [] # list of tuples -> (gt_img_name, img_size (H, W))
        num_horiz_crops = []
        num_vert_crops = []

        # load image
        for img_path, gt_path in self.img_paths:
            data_img = pil_loader(img_path)
            gt_img = pil_loader(gt_path)
            # ensure that data and gt image are of the same size
            assert gt_img.size == data_img.size
            img_names_sizes.append((os.path.basename(gt_path), data_img.size[::-1]))
            num_horiz_crops.append(math.ceil(data_img.size[0] / (self.crop_size / (1 / self.overlap))))
            num_vert_crops.append(math.ceil(data_img.size[1] / (self.crop_size / (1 / self.overlap))))

        return img_names_sizes, num_horiz_crops, num_vert_crops

    def _update_sliding_window_coordinates(self):
        self.current_horiz_crop = (self.current_horiz_crop + 1) % self.current_num_horiz_crops

        if self.current_horiz_crop == 0 or self.current_vert_crop == 0:
            self.current_vert_crop = (self.current_vert_crop + 1)

    def _get_crop_coordinates(self):
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
