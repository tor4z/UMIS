import os
import copy
import numpy as np
from .hela import HelaDataset
from .vesselNN import vesselNN

datasets = {
    'hela': HelaDataset,
    'vesslNN': vesslNN,
    'drive': DriveDataset
}


class SingleFoldValid(object):
    def __init__(self, opt):
        images_path = os.path.join(opt.image_path, '*.{}'format(opt.suffix))
        images = sorted(glob.glob(images_path))

        labels_path = os.path.join(opt.label_path, '*.{}'format(opt.suffix))
        labels = sorted(glob.glob(labels_path))

        datas = list(zip(images, labels))
        self.datas = np.random.shuffle(datas)

        self.len = len(self.images)
        self.val_index = 0

        self.dataset_cls = datasets[opt.dataset]

    def gen_dataset(self):
        while self.val_index < self.len:
            datas = copy.copy(self.datas)
            val_datas = datas.pop(self.val_index)
            train_datas = datas

            train_dataset = self.dataset_cls(opt, train_datas)
            val_dataset = self.dataset_cls(opt, val_datas)

            yield train_dataset, val_dataset
            self.val_index += 1
        return
