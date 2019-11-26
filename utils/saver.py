import os
import shutil
import torch
import glob

from .config import Config

class Saver(object):
    CONFIG_FILE = 'cfg.pth'
    BEST_FILE = 'best.pth'
    LATEST_FILE = 'latest.pth'
    CHECKPOINT_FILE = 'checkpoint_{}.pth'

    def __init__(self, opt):
        self.force = opt.force
        self.dir = os.path.join(opt.saver_dir, opt.runtime_id)
        if os.path.exists(self.dir):
            if self.force:
                shutil.rmtree(self.dir)
            else:
                raise RuntimeError('{} already exists.'.format(self.dir))
        else:
            try:
                os.mkdir(self.dir)
            except Exception as e:
                raise RuntimeError(str(e))
        
        self.best_filename = os.path.join(self.dir, self.BEST_FILE)
        self.latest_filename = os.path.join(self.dir, self.LATEST_FILE)
        self.config_filename = os.path.join(self.dir, self.CONFIG_FILE)

    def gen_checkpoint_filename(self, epoch):
        path = os.path.join(self.dir, self.CHECKPOINT_FILE.format(epoch))
        if os.path.exists(path):
            if self.force:
                os.remove(path)
            else:
                raise RuntimeError('{} already exists.'.format(path))

        return path 


    def save_checkpoint(self, state, epoch):
        pass

    def load_checkpoint(self, epoch):
        pass

    def save_latest(self, path):
        pass

    def load_latest(self):
        pass

    def save_best(self, path):
        pass

    def load_best(self):
        pass

    def save_config(self, opt):
        cfg = opt.dump()
        self.save(cfg, self.config_filename)

    def load_config(self, path=None):
        if path is None:
            path = self.config_filename
        cfg = self.load(path)
        opt = Config()
        opt.load(cfg)
        return opt

    def check_dir(self, path):
        if not os.path.exists(path):
            raise RuntimeError('{} not exists.'.format(path))

    def save(self, obj, path):
        self.check_dir(path)
        torch.save(obj, path)

    def load(self, obj, path):
        self.check_dir(path)
        torch.load(obj, path)
