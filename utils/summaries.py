import os
import numpy as np
from torchvision.utils import make_grid
from torch.utils import tensorboard


class Summary(object):
    def __init__(self, opt):
        path = os.path.join(opt.summary_dir, opt.runtime_id)
        self.dir = path
        self.writer = tensorboard.SummaryWriter(log_dir=path)

    def add_image(self, writer, image, output, global_step):
        z = np.random.randint(0, image.shape[2])

        grid_image = make_grid(image[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output', grid_image, global_step)

        output[output > 0.9] = 1
        output[output <= 0.9] = 0
        grid_image = make_grid(output[:3, :, z, :, :].clone().cpu().data, 3, normalize=True)
        writer.add_image('Output - mask', grid_image, global_step)

    def add_text(self):
        pass

    def add_scalars(self):
        pass

    def flush(self):
        self.writer.flush()
