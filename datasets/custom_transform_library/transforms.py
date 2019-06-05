import numbers
import random

try:
    import accimage
except ImportError:
    accimage = None


from torchvision.transforms import functional as F
from . import functional as F_custom

__all__ = ["Compose", "ToTensorTwinImage", "RandomTwinCrop", "ToTensorSlidingWindowCrop"]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt


class ToTensorSlidingWindowCrop(object):
    """
    Crop the data and ground truth image at the specified coordinates to the specified size and convert
    them to a tensor.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, gt, coordinates):
        """
        Args:
            img (PIL Image): Data image to be cropped and converted to tensor.
            gt (PIL Image): Ground truth image to be cropped and converted to tensor.

        Returns:
            Data tensor, gt tensor (tuple of tensors): cropped and converted images

        """
        x_position = coordinates[0]
        y_position = coordinates[1]

        return F.to_tensor(F.crop(img, x_position, y_position, self.crop_size, self.crop_size)),\
               F.to_tensor(F.crop(gt, x_position, y_position, self.crop_size, self.crop_size))


class ToTensorTwinImage(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (W x H x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, img, gt):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img), F.to_tensor(gt)


class OneHotToPixelLabelling(object):
    def __call__(self, tensor):
        return F_custom.argmax_onehot(tensor)


class OneHotEncoding(object):
    def __init__(self, class_encodings):
        self.class_encodings = class_encodings

    def __call__(self, gt):
        """
        Args:

        Returns:

        """
        return F_custom.gt_to_one_hot(gt, self.class_encodings)


class OneHotEncodingDIVAHisDB(object):
    def __init__(self, class_encodings, use_boundary_pixel=True):
        self.class_encodings = class_encodings
        self.use_boundary_pixel = use_boundary_pixel

    def __call__(self, gt):
        """
        Args:

        Returns:

        """
        return F_custom.gt_to_one_hot_hisdb(gt, self.class_encodings, self.use_boundary_pixel)


class RandomTwinCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    def __init__(self, crop_size):
            self.crop_size = crop_size

    @staticmethod
    def get_params(img_size, crop_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img_size (tuple): size of image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img_size
        th = crop_size
        tw = crop_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.get_params(img.size, self.crop_size)

        return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w)
