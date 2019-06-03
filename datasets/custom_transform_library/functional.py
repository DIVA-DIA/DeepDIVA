from __future__ import division
import torch
from sklearn.preprocessing import OneHotEncoder

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np

from torchvision.transforms import functional as F

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(F._is_pil_image(pic) or F._is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def gt_to_one_hot(matrix, class_encodings):
    """
    Convert ground truth tensor or numpy matrix to one-hot encoded matrix

    Parameters
    -------
    matrix: float tensor from to_tensor() or numpy array
        shape (C x H x W) in the range [0.0, 1.0] or shape (H x W x C) BGR
    class_encodings: list of int
        Blue channel values that encode the different classes

    Returns
    -------
    torch.LongTensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    num_classes = len(class_encodings)

    if type(matrix).__module__ == np.__name__:
        im_np = matrix[:, :, 2].astype(np.uint8)
    else:
        # TODO: ugly fix -> better to not normalize in the first place
        np_array = (matrix * 255).numpy().astype(np.uint8)
        im_np = np_array[2, :, :].astype(np.uint8)

    integer_encoded = np.array([i for i in range(num_classes)])
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(np.int8)

    np.place(im_np, im_np == 0, 1) # needed to deal with 0 fillers at the borders during testing (replace with background)
    replace_dict = {k: v for k, v in zip(class_encodings, onehot_encoded)}

    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[replace_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])]).astype(
        np.uint8)

    return torch.LongTensor(one_hot_matrix.transpose((2, 0, 1)))


def gt_to_one_hot_hisdb(matrix, class_encodings, use_boundary_pixel=True):
    """
    Convert ground truth tensor or numpy matrix to one-hot encoded matrix

    Parameters
    -------
    matrix: float tensor from to_tensor() or numpy array
        shape (C x H x W) in the range [0.0, 1.0] or shape (H x W x C) BGR
    class_encodings: List of int
        Blue channel values that encode the different classes
    use_boundary_pixel : boolean
        Use boundary pixel
    Returns
    -------
    torch.LongTensor of size [#C x H x W]
        sparse one-hot encoded multi-class matrix, where #C is the number of classes
    """
    num_classes = len(class_encodings)

    if type(matrix).__module__ == np.__name__:
        im_np = matrix[:, :, 2].astype(np.uint8)
        border_mask = matrix[:, :, 0].astype(np.uint8) != 0
    else:
        # TODO: ugly fix -> better to not normalize in the first place
        np_array = (matrix * 255).numpy().astype(np.uint8)
        im_np = np_array[2, :, :].astype(np.uint8)
        border_mask = np_array[0, :, :].astype(np.uint8) != 0

    # adjust blue channel according to border pixel in red channel -> set to background
    if use_boundary_pixel:
        im_np[border_mask] = 1

    integer_encoded = np.array([i for i in range(num_classes)])
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded).astype(np.int8)

    np.place(im_np, im_np == 0, 1) # needed to deal with 0 fillers at the borders during testing (replace with background)
    replace_dict = {k: v for k, v in zip(class_encodings, onehot_encoded)}

    # create the one hot matrix
    one_hot_matrix = np.asanyarray(
        [[replace_dict[im_np[i, j]] for j in range(im_np.shape[1])] for i in range(im_np.shape[0])]).astype(
        np.uint8)

    return torch.LongTensor(one_hot_matrix.transpose((2, 0, 1)))


def argmax_onehot(tensor):
    """
    # TODO
    """
    return torch.LongTensor(np.array(np.argmax(tensor.numpy(), axis=0)))


def annotation_to_argmax(input_shape, annotations, name_onehotindex, category_id_name):
    """
    Convert ground truth tensor to one-hot encoded matrix

    Parameters
    -------
    input_shape: tuple
        image (width, height)

    annotations: list
        annotations from the COCO dataset loaded with the pycocotools and the torchvision dataset loader

    name_onehotindex: dict
        encodes the name and id for every class with the corresponding argmax number

    category_id_name: dict
        encodes the category id and the corresponding class name

    Returns
    -------
    torch.LongTensor of size [H x W]
        argmax of the classes
    """
    from skimage.draw import polygon

    gt_img = np.zeros(input_shape, 'uint8')

    for ann in annotations:
        for seg in ann['segmentation']:
            if type(ann['segmentation']) == list:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))

                rr, cc = polygon(poly[:, 0], poly[:, 1], input_shape)
                gt_img[rr, cc] = name_onehotindex[category_id_name[ann['category_id']]]

    return torch.LongTensor(gt_img)
