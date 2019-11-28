import os
import copy
import numpy as np
import glob

from .hela import HelaDataset
from .vesselNN import VesselNN
from .drive import DriveDataset

datasets = {
    'hela': HelaDataset,
    'vesselNN': VesselNN,
    'drive': DriveDataset
}


class SingleFoldValid(object):
    def __init__(self, opt):
        images_path = os.path.join(opt.image_path, '*.{}'.format(opt.suffix))
        self.images = sorted(glob.glob(images_path))

        # if opt.labels_path:
        #     labels_path = os.path.join(opt.label_path, '*.{}'format(opt.suffix))
        #     labels = sorted(glob.glob(labels_path))
        #     datas = list(zip(images, labels))
        #     self.datas = np.random.shuffle(datas)
        # else:
        #     labels = []
        #     self.datas = np.random.shuffle(datas)

        self.image_len = len(self.images)
        # self.label_len = len(labels)
        self.val_index = 0

        self.dataset_cls = datasets[opt.dataset]
        self.opt = opt
        print('initialize Single fold validate')
        print('loaded {} image'.format(self.image_len))

    def supervised(self):
        pass
        # while self.val_index < self.image_len:
        #     datas = copy.copy(self.datas)
        #     val_datas = datas.pop(self.val_index)
        #     train_datas = datas

        #     train_dataset = self.dataset_cls(opt, train_datas)
        #     val_dataset = self.dataset_cls(opt, val_datas)

        #     yield train_dataset, val_dataset
        #     self.val_index += 1
        # return

    def unsupervised(self):
        while self.val_index < self.image_len:
            datas = copy.copy(self.images)
            val_datas = datas.pop(self.val_index)
            train_datas = datas

            train_dataset = self.dataset_cls(self.opt, train_datas)
            val_dataset = self.dataset_cls(self.opt, val_datas)

            yield train_dataset, val_dataset
            self.val_index += 1
        return

    def gen_dataset(self):
        return self.unsupervised()
        
        # if self.image_len == self.label_len:
        #     return self.supervised()
        # else:
        #     return self.unsupervised()
