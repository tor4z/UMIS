import os
import numpy as np
from torchvision.utils import make_grid
from torch.utils import tensorboard


class Summary(object):
    def __init__(self, opt):
        self.dir = os.path.join(opt.summary_dir, opt.runtime_id)
        self.writer = tensorboard.SummaryWriter(log_dir=path)
        self.thredhold = opt.thredhold
        self.z = np.random.randint(0, image.shape[2])
        self.disp_images = opt.disp_images

    def add_image(self, tag, image, global_step):
        self.writer.add_imag(tag, image, global_steps)

    def train_image(self, input, seg, rec, global_steps):
        # display input image
        grid_image = make_grid(input[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/input', grid_image, global_step)

        # display seg image
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/seg', grid_image, global_step)

        # display seg image with thredhold
        seg[seg > self.thredhold] = 1
        seg[seg <= self.thredhold] = 0
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data, 
                                self.disp_images, normalize=True)
        writer.add_image('train/seg/thredhold', grid_image, global_step)

        # display rec image
        grid_image = make_grid(rec[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        writer.add_image('train/rec', grid_image, global_step)

    def val_image(self, input, seg, global_steps):
        # display input image
        grid_image = make_grid(input[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/input', grid_image, global_step)

        # display seg image
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/seg', grid_image, global_step)

        # display seg image with thredhold
        seg[seg > self.thredhold] = 1
        seg[seg <= self.thredhold] = 0
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                    self.disp_images, normalize=True)
        writer.add_image('val/seg/thredhold', grid_image, global_step)

    def add_text(self, tag, text, global_steps):
        self.writer.add_text(tag, text, global_steps)
        self.flush()

    def add_scalars(self, tag, scalars, global_steps):
        self.writer.add_scalars(tag, scalars, global_steps)
        self.flush()

    def flush(self):
        self.writer.flush()
