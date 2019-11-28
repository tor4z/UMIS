import os
import glob
import torch
from torch.utils.data import Dataset

from .utils import imread
from .transforms import Padding, CenterCrop


class HelaDataset(Dataset):
    def __init__(self, opt, images):
        self.images = images
        # self.labels = list(unziped[1])

        self.len = len(self.images)

        self.x = opt.image_x
        self.y = opt.image_y
        self.z = opt.image_z

        self.crop = CenterCrop(opt)

    def __len__(self):
        return self.len

    def imread(self, path, tensor=True):
        image = imread(path)

        if tensor:
            return torch.Tensor(image)
        else:
            return image

    def naive_resize(self, image):
        z, y, x = image.shape

        if z < self.z or x < self.x or y < self.y:
            pad_size = (max(abs(x - self.x), abs(y - self.y), abs(z - self.z)) // 2) + 1
            pad = Padding(pad_size)
            image = pad(image)
        return self.crop(image)

    def __getitem__(self, index):
        image = self.images(index)
        # label = self.labels(index)

        image = self.naive_resize(image)
        # label = self.naive_resize(label)

        return image, None
