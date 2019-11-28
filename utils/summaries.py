import os
import numpy as np
from torchvision.utils import make_grid
from torch.utils import tensorboard


class Summary(object):
    def __init__(self, opt):
        self.dir = os.path.join(opt.summary_dir, opt.dataset, opt.runtime_id)
        self.writer = tensorboard.SummaryWriter(log_dir=self.dir)
        self.thredhold = opt.thredhold
        self.dim = opt.dim
        self.z = np.random.randint(0, opt.image_z)
        self.disp_images = opt.disp_images
        print('initialize summary')

    def add_image(self, tag, image, global_steps):
        self.writer.add_image(tag, image, global_steps)

    def auto_adapt(self, data):
        if self.dim == 2 and data is not None:
            data = data.unsqueeze(2)
        return data

    def train_image(self, input, seg, rec, label, global_steps):
        input = self.auto_adapt(input)
        seg = self.auto_adapt(seg)
        rec = self.auto_adapt(rec)
        label = self.auto_adapt(label)

        # display input image
        grid_image = make_grid(input[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/input', grid_image, global_steps)

        if label:
            # display label image
            grid_image = make_grid(label[:self.disp_images, :, self.z, :, :].data,
                                    self.disp_images, normalize=True)
            self.add_image('train/label', grid_image, global_steps)

        # display seg image
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/seg', grid_image, global_steps)

        # display seg image with thredhold
        seg[seg > self.thredhold] = 1
        seg[seg <= self.thredhold] = 0
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data, 
                                self.disp_images, normalize=True)
        self.add_image('train/seg/thredhold', grid_image, global_steps)

        # display rec image
        grid_image = make_grid(rec[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('train/rec', grid_image, global_steps)

    def val_image(self, input, seg, label, global_steps):
        input = self.auto_adapt(input)
        seg = self.auto_adapt(seg)
        label = self.auto_adapt(label)

        # display input image
        grid_image = make_grid(input[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/input', grid_image, global_steps)

        if label:
            # display label image
            grid_image = make_grid(label[:self.disp_images, :, self.z, :, :].data,
                                    self.disp_images, normalize=True)
            self.add_image('train/label', grid_image, global_steps)

        # display seg image
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                self.disp_images, normalize=True)
        self.add_image('val/seg', grid_image, global_steps)

        # display seg image with thredhold
        seg[seg > self.thredhold] = 1
        seg[seg <= self.thredhold] = 0
        grid_image = make_grid(seg[:self.disp_images, :, self.z, :, :].clone().cpu().data,
                                    self.disp_images, normalize=True)
        self.add_image('val/seg/thredhold', grid_image, global_steps)

    def add_text(self, tag, text, global_steps):
        self.writer.add_text(tag, text, global_steps)
        self.flush()

    def add_scalars(self, tag, scalars, global_steps):
        self.writer.add_scalars(tag, scalars, global_steps)
        self.flush()

    def add_scalar(self, tag, value, global_steps):
        self.writer.add_scalar(tag, value, global_steps)
        self.flush()

    def flush(self):
        self.writer.flush()
