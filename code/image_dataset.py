import collections
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
import tifffile
from pathlib import Path
import re
from msd_pytorch.errors import InputError
import logging

'''
This code was copied and adapated from:
https://github.com/ahendriksen/msd_pytorch/msd_pytorch/image_dataset.py
'''


def _natural_sort(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", x)]

    return sorted(l, key=key)


def _convert_to_integral(img):
    """Convert numpy array to integral value type.

    Handles Boolean arrays, signed and unsigned integer type arrays.

    :param img: A numpy array to convert.
    :returns:
    :rtype:

    """
    if img.dtype.kind == "u":
        return img
    elif img.dtype.kind == "i":
        logging.warning("Converting signed integer image to unsigned integer.")
        # The PyTorch segmentation losses require 64 bit labels for
        # some reason. We might as well convert the image to uint64
        # here.
        return img.astype(np.uint64)
    elif img.dtype.kind == "b":
        # convert boolean to unsigned integer.
        return img.astype(np.uint8)
    else:
        return img.astype(np.uint8) # CHANGED by J. Minnema
        #raise InputError(
        #    f"Image could not be converted to an integral value. Its type is {img.dtype}."
        #)


def _relabel_image(img, labels):
    img = _convert_to_integral(img)

    if isinstance(labels, collections.Iterable):
        # Check for values in the image that are not in the label set:
        non_labels = set(np.unique(img)) - set(labels)
        if non_labels:
            raise InputError(
                f"Encountered unexpected values {non_labels} that are not in the label set."
            )
        # Relabel the image
        data = np.copy(img)
        for i, label in enumerate(labels):
            data[img == label] = i
            return data
    else:
        # Image values should be contained in [0, labels-1]. We check this and return the image.
        if img.min() < 0 or labels <= img.max():
            raise InputError(
                f"Image pixel value range {[img.min(), img.max()]} exceeded range {[0, labels - 1]}."
            )
        else:
            return img


def _load_natural_image(path):
    img = np.array(imageio.imread(path))

    # If the image is a gray-scale, RGB, or RGBA image. The channel
    # dimension will be last. We move it to the front.
    if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
        return img.swapaxes(0, 2)
    else:
        return img


class ImageStack(object):
    """A stack of images stored on disk.

    An image stack describes a collection of images matching the
    file path specifier `path_specifier`.

    The images can be tiff files, or any other image filetype
    supported by imageio.

    The image paths are sorted using a natural sorting
    mechanism. So "scan1.tif" comes before "scan10.tif".

    Images can be retrieved by indexing into the stack. For example:

    ``ImageStack("*.tif")[i]``

    These images are returned as torch
    tensors with three dimensions CxHxW.

    """

    def __init__(self, paths, *,  collapse_channels=False, labels=None):
        """Create a new ImageStack.

        :param paths: list
        
        List that describes all image file paths. 
        
        :param collapse_channels: `bool`

        By default, the images are returned in the CxHxW format, where
        C is the number of channels and H and W specify the height and
        width, respectively.

        If `collapse_channels=True`, then all channels in the image
        will be averaged to a single channel. This can be used to
        convert color images to gray-scale images, for instance.

        If `collapse_channels=False`, any channels in the image will
        be retained.

        In either case, the returned images have at least one channel.

        :param labels: `int` or `list(int)`

        By default, all image pixel values are converted to
        float32.

        If you want to retrieve the image pixels as
        integral values instead, set

        * `labels=k` for an integer `k` if the labels are
           contained in the set {0, 1, ..., k-1};
        * `labels=[1,2,5]` if the labels are contained in the set
           {1,2,5}.

        Setting labels is useful for segmentation.

        :returns: An ImageStack
        :rtype:

        """
        super(ImageStack, self).__init__()

        self.collapse_channels = collapse_channels
        self.labels = labels
        self.paths = paths


    @property
    def num_labels(self):
        """The number of labels in this image stack.

        If the stack is not labeled, this property access raises a
        RuntimeError.

        :returns: The number of labels in this image stack.
        :rtype: int

        """
        if self.labels is None:
            raise RuntimeError("This image stack has no labels")
        elif isinstance(self.labels, collections.Iterable):
            return len(list(self.labels))
        else:
            return int(self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]

        # Load image
        try:
            if Path(path).suffix.lower() in [".tif", ".tiff"]:
                img = np.array(tifffile.imread(path))
            else:
                img = _load_natural_image(path)
        except Exception as e:
            raise InputError(f"Could not read image from {path}. Got error {e}")

        # Convert image type if necessary:
        if self.labels is not None:
            try:
                img = _relabel_image(img, self.labels)
            except InputError as e:
                raise InputError(
                    f"Expected labeled image from path {path}. Got error {e}"
                )
        else:
            img = img.astype(np.float32)

        # Check and set image dimensions
        if img.ndim > 3:
            img = np.squeeze(img) # CHANGED by J. Minnema
            #raise InputError(f"Image in {path} has more than 3 dimensions.")

        elif img.ndim < 2:
            raise InputError(f"Image in {path} has less than 2 dimensions.")

        # The # of dimensions can be 2 or 3 at this point. Make it 3.
        if img.ndim == 2:
            img = img[None, ...]

        # Collapse channels if necessary
        assert not (
            self.labels is not None and self.collapse_channels
        ), "Cannot collapse channels of segmentation image stack."

        if self.collapse_channels:
            img = np.mean(img, axis=0, keepdims=True)

        img = torch.from_numpy(img)

        return img


class ImageDataset(Dataset):
    """A dataset for images stored on disk.

    """

    def __init__(
        self,
        input_paths,  
        target_paths, 
        collapse_channels=False,
        labels=None,
    ):
        """Create a new image dataset.

        :param input_paths: list
        
        List that describes the image file paths of all input images. 
        
        :param target_paths: list
        
        List that describes the image file paths of all target images. 

        :param collapse_channels: `bool`

        By default, the images are returned in the CxHxW format,
        where C is the number of channels and H and W specify the
        height and width, respectively.

        If `collapse_channels=True`, then all channels in the
        image will be averaged to a single channel. This can be
        used to convert color images to gray-scale images, for
        instance.

        If `collapse_channels=False`, any channels in the image
        will be retained.

        In either case, the returned images have at least one
        channel.

        :param labels: `int` or `list(int)`

        By default, both input and target image pixel values are
        converted to float32.

        If you want to retrieve the target image pixels as
        integral values instead, set:

        * ``labels=k`` for an integer ``k`` if the labels are contained in the set {0, 1, ..., k-1};
        * ``labels=[1,2,5]`` if the labels are contained in the set {1,2,5}.

        Setting labels is useful for segmentation.

        :returns:
        :rtype:

        """
        super(ImageDataset, self).__init__()
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.collapse_channels = collapse_channels
        self.labels = labels

        # Do not collapse channels in the target images when we do
        # segmentation. This is not supported.
        collapse_target = collapse_channels and labels is None
        self.input_stack = ImageStack(
            input_paths, collapse_channels=collapse_channels)
        self.target_stack = ImageStack(
            target_paths, collapse_channels=collapse_target, labels=labels)

        if len(self.input_stack) != len(self.target_stack):
            raise InputError(
                f"Number of input and target images does not match. "
                f"Got {len(self.input_stack)} input images and {len(self.target_stack)} target images."
            )

    def __len__(self):
        return len(self.input_stack)

    def __getitem__(self, i):
        return (self.input_stack[i], self.target_stack[i])

    @property
    def num_labels(self):
        """The number of labels in this image stack.

        If the stack is not labeled, this property access raises a
        RuntimeError.

        :returns: The number of labels in this image stack.
        :rtype: int

        """
        return self.target_stack.num_labels
