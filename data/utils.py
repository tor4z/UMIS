from libtiff import TIFF
import imageio
import cv2
import numpy as np


def read_gif(filename):
    im = imageio.mimread(filename)
    im = np.array(im)
    return im

def read_tif(filename):
    tif = TIFF.open(filename)
    imgs = []
    for img in tif.iter_images():
        imgs.append(img)
    return np.array(imgs)


def read_jpg(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1]


def read_png(filename):
    return read_jpg(filename)


def imread(filename):
    suffix = filename.split('.')[-1]
    if suffix == 'gif':
        return read_gif(filename)
    elif suffix == 'tif':
        return read_tif(filename)
    elif suffix == 'jpg':
        return read_jpg(filename)
    elif suffix == 'png':
        return read_png(filename)
    else:
        raise NotImplementedError('Not support read {} file'.format(suffix))