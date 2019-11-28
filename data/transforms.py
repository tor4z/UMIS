from torch import nn


class Padding(object):
    def __init__(self, size):
        self.size = size
        self.pad2d = nn.ConstantPad2d(self.size, 0)
        self.pad3d = nn.ConstantPad3d(self.size, 0)

    def pad_2d(self, image):
        return self.pad2d(image)

    def pad_3d(self, image):
        return self.pad3d(image)

    def __call__(self, image):
        if len(image.shape) == 2:
            return self.pad2d(image)
        else:
            return self.pad3d(image)


class CenterCrop(object):
    def __init__(self, opt):
        self.x = opt.image_x
        self.y = opt.image_y
        self.z = opt.image_z

    def crop_2d(self, image):
        _, y, x = image.size()
        if x > self.x or y > self.y:
            raise ValueError('({},{},{}) crop to ({},{},{})'.format(\
                                x, y, self.x, self.y))

        pad_x = (x - self.x) // 2
        pad_y = (y - self.y) // 2

        return image[pad_y: pad_y + self.y,
                     pad_x: pad_x + self.x,]

    def crop_3d(self, image):
        z, y, x = image.shape
        if x > self.x or y > self.y or z > self.z:
            raise ValueError('({},{},{}) crop to ({},{},{})'.format(\
                                x, y, z, self.x, self.y, self.z))

        pad_x = (x - self.x) // 2
        pad_y = (y - self.y) // 2
        pad_z = (z - self.z) // 2

        return image[pad_z: pad_z + self.z,
                     pad_y: pad_y + self.y,
                     pad_x: pad_x + self.x,]

    def __call__(self, image):
        if self.z == 1:
            return self.crop_2d(image)
        else:
            return self.crop_3d(image)


class Resize(object):
    def __init__(self, opt):
        pass

    def __call__(self):
        pass
