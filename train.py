from torch.utils.data import DataLoader

from utils.config import Config
from utils.saver import Saver
from utils.summaries import Summary
from models.trainner import Trainner
from data import get_dataset

from models.segnet import SegNet2D

CFG_FILE = 'cfg.yaml'


def main(opt):
    saver = Saver(opt)
    summary = Summary(opt)
    dataset = get_dataset(opt)
    trainner = Trainner(opt)
    dataloader = DataLoader(dataset, 
                            batch_size=opt.batch_size,
                            shuffle=opt.shuffle,
                            num_workers=opt.num_workers)
    trainner.set_data(dataloader)


if if __name__ == "__main__":
    opt = Config(CFG_FILE)
    main(opt)