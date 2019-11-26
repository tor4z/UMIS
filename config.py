import yaml
import datetime
import torch


class Config(object):
    def __init__(self, cfg):
        with open(cfg) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.init_attr()
        self.post_config()

    def init_attr(self):
        for k, v in self.cfg.items():
            self.__setattr__(k, v)

    def post_config(self):
        if self.devices:
            self.cuda= True
            self.device = torch.device('cuda:{}'.format(self.devices[0]))
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.resume and not self.runtime_id:
            raise RuntimeError('when resume is enabled, runtime_id should be specified.')

        if not self.runtime_id:
            dtate_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.runtime_id = dtate_str


'''
def get_configs():
    # hyper parameter
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=1000, metavar='N', help='Number of epochs')
    args.add_argument('--batch_size', type=int, default=64, metavar='N', help='#CUDA * batch_size')
    args.add_argument('--lr', type=float, default=1e-4, metavar='N', help='Learning rate')
    args.add_argument('--lmd1', type=int, default=1, metavar='N', help='lambda 1')
    args.add_argument('--lmd2', type=int, default=2, metavar='N', help='lambda 2')
    args.add_argument('--range_norm', action='store_true', help='range-norm')

    # dataset
    args.add_argument('--train-images-path', type=str, metavar='N', help='Training dataset images path')
    args.add_argument('--train-labels-path', type=str, metavar='N', help='Training dataset labels path')
    args.add_argument('--val-image-path', type=str, metavar='N', help='Validation image path')
    args.add_argument('--val-label-path', type=str, metavar='N', help='Validation label path')
    args.add_argument('--validate', action='store_true', help='validatargs')

    # morphological
    args.add_argument('--smooth_iter', type=int, default=3, help='ACWE Smooth')

    # set prepare work
    args.add_argument('--summary_dir', type=str, default='runs', help='summary output dir')
    args.add_argument('--saver_dir', type=str, default='storage', help='saved model')
    args.add_argument('--workers', type=int, default=3, help='Dataload workers')
    args.add_argument('--devices', nargs='+', type=int, default=None, help='GPU IDs')
    args.add_argument('--resume', action='store_false', help='resume')
    args.add_argument('--runtime_id', type=str, default='', help='runtime id')
    args.add_argument('-f', '--force', action='store_false', help='remove old data and run')
    
    # image settings
    args.add_argument('--image_x', type=int, help='remove old data and run')

    return args


def post_config(opt):
    if opt.devices:
        opt.cuda= True
        opt.device = torch.device('cuda:{}'.format(opt.devices[0]))
    else:
        opt.cuda = False
        opt.device = torch.device('cpu')

    if opt.resume and not opt.runtime_id:
        raise RuntimeError('when resume is enabled, runtime_id should be specified.')

    if not opt.runtime_id:
        dtate_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        opt.runtime_id = dtate_str
    
    return opt
'''