import os
import struct
import numpy as np


def load_mnist(path, split="train"):
    """Load MNIST data from `path`.

    The code was taken from https://notebook.community/rasbt/pattern_classification/data_collecting/reading_mnist.
    """
    if split == "train":
        labels_path = os.path.join(path, "train-labels-idx1-ubyte")
        images_path = os.path.join(path, "train-images-idx3-ubyte")
    elif split == "test":
        labels_path = os.path.join(path, "t10k-labels-idx1-ubyte")
        images_path = os.path.join(path, "t10k-images-idx3-ubyte")
    else:
        raise AttributeError('`split` must be "train" or "test"')

    with open(labels_path, "rb") as lbpath:
        _ = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        _ = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
